"""Contains the class Optimade Extractor."""

import contextlib
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from math import gcd
from urllib.parse import urlparse

from optimade.client import OptimadeClient

from matcollect.core.utils.pymatgen_helper import convert_to_structure


class OptimadeExtractor:
    """Extract materials using Optimade and process into standard formats."""

    def __init__(self,
                 include_providers: None | list[str] = None,
                 exclude_providers: None | list[str] = ["aflow"],
                 max_results_per_provider: None | int = None,
                 use_async: bool = False,
                 timeout: float | None = 120.0):
        """Initialize the Optimade Extractor and connect to the Optimade API.

        Args:
            include_providers (None | list[str], optional). Defaults to None.
            exclude_providers (None | list[str], optional). Defaults to ["aflow"].
            use_async (bool, optional). Defaults to False.
            timeout (float | None, optional). Defaults to 120.0.
        """
        self.client = OptimadeClient(include_providers=include_providers,
                                     exclude_providers=exclude_providers,
                                     max_results_per_provider=max_results_per_provider,
                                     use_async=use_async,
                                     http_timeout=timeout)

        self.max_results_per_provider = max_results_per_provider
        self.extracted_materials = []
        self.found_databases = []
        self.summary = None
        self.optimade_filter = None
        self.important_tags = {
            "total_energy": [
                "_alexandria_energy_corrected",
                "_mcloud_total_energy"
                ],
            "formation_energy_per_atom": [
                "_alexandria_formation_energy_per_atom",
                "_oqmd_delta_e",
                "_odbx_formation_energy",
                "_mp_stability"
                ],
            "band_gap": [
                "_alexandria_band_gap",
                "_mcloud_band_gap",
                "_oqmd_band_gap",
                "_twodmatpedia_band_gap"
                ],
            "magnetic_moments": [
                "_alexandria_magnetic_moments"
                ]
            }

        # Store constructor kwargs so per-thread workers can create fresh instances.
        self._init_kwargs = {
            "include_providers": include_providers,
            "exclude_providers": exclude_providers,
            "max_results_per_provider": max_results_per_provider,
            "use_async": use_async,
            "timeout": timeout,
        }

    # Multi-chemsys parallel extraction
    def extract_many(
        self,
        chemsys_list: list[list[str]],
        only_elements: bool = True,
        min_elements: None | int = None,
        max_elements: None | int = None,
        max_workers: int = 8,
        ) -> dict:
        """Extract materials for multiple chemical systems in parallel.

        Each chemical system is fetched in its own thread, each with a
        dedicated :class:`OptimadeExtractor` instance to avoid shared state.

        Args:
            chemsys_list (list[list[str]]):
                A list of chemical systems, where each system is a list of
                element symbols, e.g. ``[["Fe", "O"], ["Li", "Fe", "O"]]``.
            only_elements (bool, optional):
                Passed to :meth:`generate_elements_filter`. Defaults to True.
            min_elements (None | int, optional):
                Passed to :meth:`generate_elements_filter`. Defaults to None.
            max_elements (None | int, optional):
                Passed to :meth:`generate_elements_filter`. Defaults to None.
            max_workers (int, optional):
                Maximum number of parallel threads. Defaults to 8.

        Returns:
            dict: Merged :meth:`dump` output keyed by chemsys string, with the
            same nested structure as a single :meth:`dump` call
            (i.e. ``{chemsys: {database: {material_id: attributes}}}``).
        """
        merged: dict = {}
        self._parallel_errors: list[dict] = []  # Collect errors from all workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_chemsys_worker,
                    elements,
                    only_elements,
                    min_elements,
                    max_elements,
                ): elements
                for elements in chemsys_list
            }

            for future in as_completed(futures):
                elements = futures[future]
                try:
                    result, errors = future.result()
                    self._parallel_errors.extend(errors)
                    for chemsys_key, databases in result.items():
                        merged.setdefault(chemsys_key, {})
                        for db, materials in databases.items():
                            merged[chemsys_key].setdefault(db, {})
                            merged[chemsys_key][db].update(materials)
                except Exception as exc:
                    chemsys_str = "-".join(elements)
                    raise RuntimeError(
                        f"Failed to extract chemsys '{chemsys_str}': {exc}"
                    ) from exc

        return merged


    def _extract_chemsys_worker(
        self,
        elements: list[str],
        only_elements: bool,
        min_elements: None | int,
        max_elements: None | int,
    ) -> tuple[dict, list[dict]]:
        extractor = OptimadeExtractor(**self._init_kwargs)
        extractor.generate_elements_filter(
            elements,
            only_elements=only_elements,
            min_elements=min_elements,
            max_elements=max_elements,
        )
        extractor.extract(flush=True)
        errors = extractor.parse_errors()
        return extractor.dump(), errors

    def generate_elements_filter(self, elements: list[str],
                                 only_elements: bool = True,
                                 min_elements: None | int = None,
                                 max_elements: None | int = None) -> str:
        """
        Generate a filter for the Optimade API based on the given elements.

        Args:
            elements (list[str]): The list of elements to filter by.
            only_elements (bool, optional):
                If True, filter by materials with only the given elements. Defaults to True.
            min_elements (None | int, optional):
                The minimum number of elements to filter by. Defaults to None.
            max_elements (None | int, optional):
                The maximum number of elements to filter by. Defaults to None.

        Returns:
            str: The filter string for the Optimade API.
        """
        # Type checking
        if not isinstance(elements, list):
            raise TypeError("elements must be a list of strings")
        if not all(isinstance(element, str) for element in elements):
            raise TypeError("elements must be a list of strings")

        # Generate the filter
        optimade_filter = f'elements HAS "{elements[0]}"'
        for element in elements[1:]:
            optimade_filter += f' AND elements HAS "{element}"'

        if only_elements:
            optimade_filter += f" AND elements LENGTH {len(elements)}"
        # Range of elements: use OR-equalities
        elif min_elements is not None or max_elements is not None:
            min_e = min_elements if min_elements is not None else 1
            max_e = max_elements if max_elements is not None else len(
                elements)
            length_filters = [
                f"elements LENGTH {i}" for i in range(min_e, max_e + 1)]
            range_filter = " OR ".join(length_filters)
            optimade_filter += f" AND ({range_filter})"

        self.optimade_filter = optimade_filter
        return self.optimade_filter

    def parse_errors(self) -> list[dict]:
        """
        Parse the errors from the Optimade client into a list of dictionaries
        containing the provider, status code, and friendly error message.

        Returns:
            list[dict]: A list of dictionaries containing the parsed errors.
        """
        # Return aggregated parallel errors if extract_many was used
        if hasattr(self, "_parallel_errors"):
            return self._parallel_errors

        errors = []
        for endpoint_results in self.client.all_results.values():
            for filter_results in endpoint_results.values():
                for base_url, query_results in filter_results.items():
                    if query_results.errors:
                        for error in query_results.errors:
                            parsed = self._pretty_error(base_url, error)
                            errors.append(parsed)
        return errors

    def _pretty_error(self, base_url: str, error: str) -> dict:

        match = re.match(r"RuntimeError: (\d+) - (https?://\S+?):", error)
        status_code = match.group(1) if match else "Error"

        # Extract just the provider hostname
        provider = urlparse(base_url).netloc or base_url

        messages = {
            "502": "Bad gateway — server temporarily unavailable",
            "404": "Not found — endpoint does not exist",
            "503": "Service unavailable",
            "500": "Internal server error",
            "429": "Too many requests — rate limited",
        }
        friendly = messages.get(status_code, "Unexpected error")
        return {"status": status_code, "provider": provider, "friendly": friendly}


    def extract(self, flush: bool = True) -> dict:
        """
        Extract materials from the Optimade API based on the generated filter.

        Args:
            flush (bool, optional):
                If True, flush the extracted materials, found databases, and summary.
                Defaults to True.

        Returns:
            dict: The extracted materials in the Optimade API format.
        """
        # Flush the extracted materials
        if flush:
            self.extracted_materials = []
            self.found_databases = []
            self.summary = None

        # Get the materials asyncronously
        self.extracted_materials = self.client.get(self.optimade_filter)

        # Get the found databases
        self.found_databases = list(
            self.extracted_materials["structures"][self.optimade_filter].keys())
        self.found_databases.sort()
        self._process_extracted_materials()  # Standardizes some important information
        return self.extracted_materials

    def _process_extracted_materials(self):
        """
        Process the extracted materials from the Optimade API into standard formats.
        This includes setting the chemical formula, reduced formula, chemsys, number of elements,
        composition, reduced composition, fractional composition, space group symbol,
        space group number, total energy, formation energy per atom, band gap, magnetic moments.
        """
        extracted_materials = self.extracted_materials["structures"][self.optimade_filter]
        # Process the extracted materials
        for database in self.found_databases:
            for material in extracted_materials[database]["data"]:
                source_attributes = material["attributes"]
                normalized_attributes = {}

                # Unmodified attributes
                normalized_attributes["material_id"] = str(material.get("id", None))
                normalized_attributes["database"] = str(database)
                normalized_attributes["nsites"] = source_attributes.get("nsites", None)
                normalized_attributes["lattice_vectors"] = source_attributes.get(
                    "lattice_vectors", None)
                normalized_attributes["cartesian_site_positions"] = source_attributes.get(
                    "cartesian_site_positions", None)
                normalized_attributes["species_at_sites"] = source_attributes.get(
                    "species_at_sites", None)
                normalized_attributes["nperiodic_dimensions"] = source_attributes.get(
                    "nperiodic_dimensions", None)

                # Check if the material has evert
                material_has_structure_info = all([
                    "lattice_vectors" in source_attributes,
                    "cartesian_site_positions" in source_attributes,
                    "species_at_sites" in source_attributes
                    ])

                # Check if the material has a structure
                if material_has_structure_info:
                    structure_attributes = self._get_attributes_from_structure(source_attributes)
                else:
                    # If the material does not have a structure, get info from the attributes
                    structure_attributes = self._get_attributes_from_databases(source_attributes)
                normalized_attributes.update(structure_attributes)

                # These are obtained from the dictionary regardless of structure availability
                normalized_attributes["total_energy"] = self._get_total_energy(source_attributes)
                e_formation = self._get_formation_energy_per_atom(source_attributes, database)
                normalized_attributes["formation_energy_per_atom"] = e_formation
                normalized_attributes["band_gap"] = self._get_band_gap(source_attributes)
                moments = self._get_magnetic_moments(source_attributes)
                normalized_attributes["magnetic_moments"] = moments

                # Update the material attributes
                material["normalized_attributes"] = normalized_attributes

    def dump(self) -> dict:
        """
        Dump the extracted materials into a nested dictionary format.

        Returns a dictionary of the following format:

        {
            chemsys: {
                database: {
                    material_id: {
                        attribute_name: attribute_value,
                        ...
                    },
                    ...
                },
                ...
            },
            ...
        }

        :return: A nested dictionary of extracted materials.
        :rtype: dict
        """
        export_materials = {}
        extracted_materials = self.extracted_materials["structures"][self.optimade_filter]
        for database in self.found_databases:
            for database_count, material in enumerate(extracted_materials[database]["data"]):
                # Explicitly limit the number of results per provider
                if (self.max_results_per_provider is not None
                    and database_count >= self.max_results_per_provider):
                    break

                source_attributes = material["attributes"]
                normalized_attributes = material["normalized_attributes"]
                chemsys = str(normalized_attributes["chemsys"])
                material_id = str(normalized_attributes["material_id"])

                export_materials.setdefault(chemsys, {})
                export_materials[chemsys].setdefault(database, {})
                export_materials[chemsys][database].setdefault(material_id, {})
                material_dict = export_materials[chemsys][database][material_id]
                material_dict.setdefault("source_attributes", {})
                material_dict.setdefault("normalized_attributes", {})
                material_dict["source_attributes"].update(source_attributes)
                material_dict["normalized_attributes"].update(normalized_attributes)
        return export_materials

    def set_filter(self, optimade_filter: list[str] | str | None) -> str:
        """
        Set the Optimade filter for the OptimadeExtractor.

        Args:
            optimade_filter (list[str] | str | None):
                The filter to set. Can be a list of strings or a string.
                If a list is provided, it will be joined with " AND ".
                If None is provided, the filter will be set to None.

        Returns:
            str: The set filter.
        """
        if optimade_filter is not None:
            if not isinstance(optimade_filter, (list, str)):
                raise TypeError("filter must be a list of strings or a string")
            if not all(isinstance(element, str) for element in optimade_filter):
                raise TypeError("filter must be a list of strings or a string")

        # If a list is provided, join the list with AND
        if isinstance(optimade_filter, list):
            self.optimade_filter = ' AND '.join(optimade_filter)
        else:
            self.optimade_filter = optimade_filter
        return self.optimade_filter

    def _get_attributes_from_structure(self, attributes: dict) -> dict:
        """
        Get structure information from PyMatGen structure.

        Args:
            attributes (dict):
                The attributes of the material.

        Returns:
            dict: The attributes dict with structure information.
        """
        # Load structure
        try:
            structure = convert_to_structure(attributes)
        except ValueError:
            # If the structure cannot be loaded, get info from the attributes
            return self._get_attributes_from_databases(attributes)
        new_attributes = {}

        # Set formulas
        new_attributes["chemsys"] = str(structure.chemical_system)
        new_attributes["chemical_formula"] = str(structure.formula)
        new_attributes["reduced_formula"] = str(structure.reduced_formula)

        # Set composition
        comp = structure.composition
        frac_comp = comp.fractional_composition
        new_attributes["composition"] = comp.as_dict()
        new_attributes["composition_reduced"] = comp.as_reduced_dict()
        new_attributes["composition_fractional"] = frac_comp.as_dict()
        new_attributes["nelements"] = int(structure.n_elems)
        new_attributes["elements_ratios"] = list(frac_comp.values())

        # Set space group
        spc_symbol, spc_number = structure.get_space_group_info(symprec=0.1)
        new_attributes["space_group_symbol"] = spc_symbol
        new_attributes["space_group_number"] = spc_number
        return new_attributes

    def _get_attributes_from_databases(self, attributes: dict) -> dict:
        """
        Get structure information from the attributes dictionary retrieved from the databases.

        Args:
            attributes (dict):
                The attributes of the material.

        Returns:
            dict: The attributes dict with structure information.
        """
        # New dictionary
        new_attributes = {}
        # Set the chemical formula
        new_attributes["chemical_formula"] = attributes.get("chemical_formula_descriptive")

        # Set reduced formula
        new_attributes["reduced_formula"] = attributes.get("chemical_formula_reduced")

        # Set chemsys
        new_attributes["chemsys"] = self._parse_chemsys(attributes)

        # Set number of elements
        if "nelements" not in attributes:
            try:
                new_attributes["nelements"] = len(attributes["elements"])
            except KeyError:
                new_attributes["nelements"] = None

        # Set the space group to None if structure is not available
        new_attributes["space_group_symbol"] = None
        new_attributes["space_group_number"] = None

        # Set the composition and element rations
        comp_dict = self._parse_composition(attributes)
        new_attributes.update(comp_dict)
        return new_attributes

    def _parse_chemsys(self, attributes: dict) -> str:
        """
        Parse the chemical system from the attributes dictionary retrieved from the databases.

        Args:
            attributes (dict):
                The attributes of the material.

        Returns:
            str: The parsed chemical system.
        """
        reduced_formula = attributes.get("reduced_formula")
        chemical_formula = attributes.get("chemical_formula")

        if reduced_formula is not None:
            chemsys = '-'.join(sorted(re.findall(r'[A-Z][a-z]?', reduced_formula)))
        elif chemical_formula is not None:
            chemsys = '-'.join(sorted(re.findall(r'[A-Z][a-z]?', chemical_formula)))
        else:
            elements = attributes.get("elements")
            chemsys = '-'.join(sorted(elements)) if elements else None
        return chemsys

    def _parse_composition(self, attributes: dict) -> tuple[dict, dict, dict, list]:
        """
        Parse the composition from the attributes dictionary retrieved from the databases.

        Args:
            attributes (dict):
                The attributes of the material.

        Returns:
            tuple: A tuple containing the parsed composition (dict),
                reduced composition (dict),
                fractional composition (dict),
                and element ratios (list).

        Raises:
            KeyError: If required keys are not in the attributes dictionary.
            TypeError: If the values in the attributes dictionary are not of the correct type.
            ValueError: If the values in the attributes dictionary are not valid.
            ZeroDivisionError: If the total number of elements is zero.
        """
        try:
            if attributes.get("species_at_sites"):
                counts = Counter(attributes["species_at_sites"])
                total = sum(counts.values())
            elif ("elements" in attributes
                and "element_ratios" in attributes
                and attributes["element_ratios"] is not None
                and "nelements" in attributes
                ):
                elements = attributes["elements"]
                ratios = attributes["element_ratios"]
                total = attributes["nelements"]
                counts = {
                    el: max(1, round(total * r))
                    for el, r in zip(elements, ratios, strict=False)
                }
            else:
                # Neither branch had enough data
                raise ValueError("No composition data available")  # noqa: TRY301

            greatest_divisor = reduce(gcd, counts.values())
            reduced_counts = {k: v // greatest_divisor for k, v in counts.items()}
            fractional_counts = {k: v / total for k, v in counts.items()}
            elements_ratios = [fractional_counts[e] for e in counts]

            composition = counts
            composition_reduced = reduced_counts
            composition_fractional = fractional_counts

        except (KeyError, TypeError, ValueError, ZeroDivisionError, UnboundLocalError):
            composition = None
            composition_reduced = None
            composition_fractional = None
            elements_ratios = None

        return {
            "composition": composition,
            "composition_reduced": composition_reduced,
            "composition_fractional": composition_fractional,
            "elements_ratios": elements_ratios
        }

    def _get_total_energy(self, attributes: dict) -> float | None:
        total_energy = None
        for tag in self.important_tags["total_energy"]:
            value = attributes.get(tag)
            if value is not None:
                with contextlib.suppress(TypeError, ValueError):
                    total_energy = float(value)
        return total_energy

    def _get_formation_energy_per_atom(self, attributes: dict, database: str) -> float | None:
        formation_energy_per_atom = None
        for tag in self.important_tags["formation_energy_per_atom"]:
            if tag == "_mp_stability" and "materialsproject" in database:
                if tag not in attributes or not attributes[tag]:
                    continue
                try:
                    formation_energy_per_atom = float(
                        attributes[tag]["gga_gga+u"]["formation_energy_per_atom"])
                except KeyError:
                    method_name = next(iter(attributes[tag].keys()))
                    with contextlib.suppress(KeyError, TypeError, ValueError):
                        formation_energy_per_atom = float(
                            attributes[tag][method_name]["formation_energy_per_atom"])
            else:
                value = attributes.get(tag)
                if value is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        formation_energy_per_atom = float(value)
        return formation_energy_per_atom

    def _get_band_gap(self, attributes: dict) -> float | None:
        band_gap = None
        for tag in self.important_tags["band_gap"]:
            value = attributes.get(tag)
            if value is not None:
                with contextlib.suppress(TypeError, ValueError):
                    band_gap = float(value)
        return band_gap


    def _get_magnetic_moments(self, attributes: dict) -> list[int] | None:
        magnetic_moments = None
        for tag in self.important_tags["magnetic_moments"]:
            value = attributes.get(tag)
            if value is not None:
                with contextlib.suppress(TypeError, ValueError):
                    magnetic_moments = list(value)
        return magnetic_moments
