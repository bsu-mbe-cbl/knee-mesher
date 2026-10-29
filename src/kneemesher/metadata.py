"""Metadata classes and interfaces for the Kneemesher project.

This module defines a hierarchy of metadata classes for representing and managing
subject, joint, and tissue metadata in the Kneemesher project. It includes:

- Abstract base class `MetaDataInterface` for metadata objects
- Concrete classes for subject, joint, and tissue metadata
- Enum types for tissue, cartilage, and surgery status
- PyVista visualization metadata container

The metadata classes support user-defined metadata dictionaries, unique identifier
generation, and robust type validation. Metadata is designed to be composable and
extensible for use in mesh processing and analysis workflows.
"""

import abc
import re
from datetime import datetime
from enum import Enum, unique

import pyvista


@unique
class Side(str, Enum):
    """Enum for anatomical sides.

    Represents the left, right, and center sides of a structure.
    """

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


@unique
class MidLine(str, Enum):
    """Enum for midline positions.

    Represents the medial and lateral positions relative to the midline.
    """

    MEDIAL = "medial"
    LATERAL = "lateral"
    CENTER = "center"


@unique
class Transverse(str, Enum):
    """Enum for transverse positions.

    Represents the superior and inferior positions in the transverse plane.
    """

    SUPERIOR = "superior"
    INFERIOR = "inferior"
    CENTER = "center"


@unique
class Dorsal(str, Enum):
    """Enum for dorsal positions.

    Represents the anterior and posterior positions in the dorsal plane.
    """

    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    CENTER = "center"


@unique
class RelativePosition(str, Enum):
    """Enum for relative positions.

    Represents the proximal and distal positions relative to a reference point.
    """

    PROXIMAL = "proximal"
    DISTAL = "distal"


@unique
class TissueType(str, Enum):
    """Enum for mesh types."""

    ALL = "all"
    BONE = "bone"
    CARTILAGE = "cartilage"


@unique
class CartilageType(str, Enum):
    """Enum for cartilage types."""

    ALL = "all"
    ARTICULAR = "articular"
    VERMIFORM = "vermiform"  # "Worm-like" cartilage, (generalized meniscus shape)


@unique
class SurgeryStatus(str, Enum):
    """Enum for surgery status."""

    PRE = "pre"
    DURING = "during"
    POST = "post"


@unique
class TissueProxy(str, Enum):
    """Enum for Pyvista tissue representation types."""

    MESH = "mesh"
    RECONSTRUCTION = "reconstruction"
    SEGMENTATION = "segmentation"


class MetaContainer:
    """Container for dynamically registered metadata fields with validation."""

    def __init__(self, **fields):
        """Initialize the MetaContainer with dynamic metadata fields.

        Args:
            **fields: Arbitrary keyword arguments representing metadata fields and their values.

        """
        self._fields = {}
        for name, value in fields.items():
            self.set_field(name, value)

    def set_field(self, name, value, descriptor=None):
        """Register a new field (optionally with a descriptor) and set its value."""
        if descriptor is not None:
            # Attach descriptor to this instance's class
            setattr(self.__class__, name, descriptor)
        # Use descriptor validation if present
        descriptor = getattr(self.__class__, name, None)
        setattr(self, name, value)
        if isinstance(descriptor, Validated):
            value = descriptor.validate(name, value)
        self._fields[name] = value

    def get_field(self, name):
        """Get a metadata field value by name.

        Args:
            name (str): The field name to retrieve.

        Returns:
            The value associated with `name`.

        Raises:
            KeyError: If the field is not present in the container.
        
        """
        return self._fields[name]

    def as_dict(self):
        """Return a shallow copy of the metadata fields as a dictionary.

        This is useful for serialisation or for creating combined/merged
        metadata dictionaries without exposing the internal storage
        directly.
        """
        return dict(self._fields)

    def __getattr__(self, name):
        """Provide attribute-style access to registered metadata fields.

        If the requested attribute name corresponds to a registered metadata
        field, its value is returned. Otherwise an AttributeError is raised
        to match normal attribute access semantics.
        """
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"{self.__class__.__name__} has no field '{name}'")

    def __setattr__(self, name, value):
        """Set attribute values and route updates to registered fields.

        Setting an attribute whose name matches a registered metadata field
        will invoke the field validation/setting flow (`set_field`). Other
        attributes are set on the instance as normal. Internal attributes
        such as `_fields` are always set directly.

        """
        if name in ("_fields",):
            super().__setattr__(name, value)
        elif name in self._fields:
            self.set_field(name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        """Return a string representation of the metadata fields."""
        fields_repr = ", ".join(f"{k}={v!r}" for k, v in self._fields.items())
        return f"{self.__class__.__name__}({fields_repr})"


def with_metadata(
    meta_cls=MetaContainer,
    overwrite_existing: bool = False,
    proxy: bool = True,
    container_name: str | None = None,
    **dynamic_fields,
):
    """Decorate a class with one (or more) metadata containers.

    This decorator may be applied repeatedly to the same class to register
    multiple named metadata containers. Each decoration creates a distinct
    property on the class (by default the property name is derived from
    ``meta_cls``; for the original behaviour the default primary container
    name remains ``metadata``).

    Args:
        meta_cls: The metadata container class to instantiate (default
            ``MetaContainer``).
        overwrite_existing: If True, newly provided metadata replaces any
            existing metadata on the instance. If False, fields are merged.
        proxy: If True (default) installs proxying methods so attribute lookups
            and ``dir()`` include metadata fields from all registered
            containers. If False, no proxying is attached.
        container_name: Optional explicit name for this metadata container on
            the decorated class (e.g. ``pv_metadata``). If omitted a name is
            derived from ``meta_cls``; for the base MetaContainer the property
            name ``metadata`` is used for backwards compatibility.
        **dynamic_fields: Optional descriptor fields to register on ``meta_cls``.

    The decorator registers the container in ``cls._metadata_containers`` as a
    list of (property_name, private_attr_name). ``__getattr__`` and ``__dir__``
    (when proxying is enabled) will search across all registered containers.
    If multiple containers expose the same field name, attribute access will
    raise an AttributeError describing the ambiguity and listing available
    container properties so callers can qualify the lookup (e.g.
    ``obj.pv_metadata.color``).

    """

    def decorator(cls):
        # record existing wrappers if present
        orig_init = getattr(cls, "__init__")
        orig_repr = getattr(cls, "__repr__")
        orig_str = getattr(cls, "__str__")
        orig_dir = getattr(cls, "__dir__")
        orig_attr = getattr(cls, "__getattr__", None)

        # ensure a registry on the class to track all decorated metadata
        containers = getattr(cls, "_metadata_containers", [])

        # choose a property name for this container
        if container_name is None:
            if meta_cls is MetaContainer:
                prop_name = "metadata"
            else:
                prop_name = meta_cls.__name__.lower()
        else:
            prop_name = container_name

        # private attribute name stored on the instance (double-underscore
        # so Python mangles it per-class, reducing accidental external access)
        private_name = f"__{prop_name}"

        # avoid duplicate registration
        if any(pn == prop_name for pn, _ in containers):
            # already registered: just return the class unchanged
            return cls

        def __init__(self, *args, metadata=None, **kwargs):
            # initialize only this container's private storage
            setattr(self, private_name, None)
            meta_kwargs = {k: kwargs.pop(k) for k in dynamic_fields.keys() & kwargs.keys()}

            # call the existing initializer (may be previously wrapped)
            orig_init(self, *args, **kwargs)

            # create the container instance for this decoration
            if metadata is not None:
                new_metadata = metadata
            elif meta_cls and dynamic_fields:
                DynamicMeta = type(
                    f"Dynamic{meta_cls.__name__}",
                    (meta_cls,),
                    {k: v for k, v in dynamic_fields.items()},
                )
                new_metadata = DynamicMeta(**meta_kwargs)
            elif meta_cls:
                new_metadata = meta_cls(**meta_kwargs)
            else:
                new_metadata = None

            # combine with existing container value if present
            existing = getattr(self, private_name, None)
            if not overwrite_existing and existing is not None:
                combined_fields = dict(existing.as_dict())
                if new_metadata:
                    combined_fields.update(new_metadata.as_dict())
                meta_class = type(new_metadata) if new_metadata else type(existing)
                setattr(self, private_name, meta_class(**combined_fields))
            else:
                setattr(self, private_name, new_metadata)

        def make_property(pn, priv):
            def _get(self):
                val = getattr(self, priv)
                if val is None:
                    raise AttributeError(f"{cls.__name__} has no '{pn}' metadata set. Access via: {', '.join(n for n, _ in getattr(cls, '_metadata_containers', []))}")
                return val

            return property(_get)

        def __getattr__(self, name):
            """Search all registered metadata containers for `name`.

            If the attribute is present in exactly one container, return it.
            If it is present in multiple containers, raise a helpful
            AttributeError instructing the caller to qualify the lookup.
            """
            matches = []
            for pn, priv in getattr(cls, "_metadata_containers", []):
                try:
                    cont = getattr(self, priv)
                except Exception:
                    cont = None
                if cont is None:
                    continue
                try:
                    val = getattr(cont, name)
                except AttributeError:
                    continue
                else:
                    matches.append((pn, val))

            if len(matches) == 1:
                return matches[0][1]
            if len(matches) > 1:
                names = ", ".join(p for p, _ in matches)
                raise AttributeError(
                    f"Ambiguous metadata attribute '{name}' found in multiple metadata containers: {names}. "
                    f"Qualify the lookup via one of: {', '.join(p for p, _ in getattr(cls, '_metadata_containers', []))}"
                )

            # fall back to original getattr if present
            if orig_attr is not None:
                return orig_attr(self, name)
            raise AttributeError(f"{cls.__name__} has no attribute '{name}'")

        def __repr__(self):
            """Include all registered metadata containers in the repr."""
            parts = [orig_repr(self)[:-1]]
            for pn, priv in getattr(cls, "_metadata_containers", []):
                try:
                    cont = getattr(self, priv)
                except Exception:
                    cont = None
                if cont:
                    parts.append(f"{pn}={cont!r}")
            return ", ".join(parts) + ")"

        def __str__(self):
            parts = [orig_str(self)]
            for pn, priv in getattr(cls, "_metadata_containers", []):
                try:
                    cont = getattr(self, priv)
                except Exception:
                    cont = None
                if cont:
                    parts.append(f", {pn}={cont}")
            return "".join(parts)

        def __dir__(self):
            base_dir = orig_dir(self)
            all_fields = set(base_dir)
            for pn, priv in getattr(cls, "_metadata_containers", []):
                try:
                    cont = getattr(self, priv)
                except Exception:
                    cont = None
                if cont:
                    all_fields.update(getattr(cont, "_fields", {}).keys())
            return sorted(all_fields)

        # Attach the init wrapper and the new property
        cls.__init__ = __init__
        # register this container on the class
        containers.append((prop_name, private_name))
        cls._metadata_containers = containers

        # attach proxying methods only when requested
        if proxy:
            cls.__getattr__ = __getattr__
            cls.__dir__ = __dir__

        # attach nicer repr/str
        cls.__repr__ = __repr__
        cls.__str__ = __str__
        # install the property for direct access
        setattr(cls, prop_name, make_property(prop_name, private_name))

        return cls

    return decorator


class Validated(abc.ABC):
    """Abstract descriptor base for validated metadata fields.

    Subclasses must implement `validate(name, value)` to enforce constraints
    on values assigned to metadata descriptors. This base class implements
    the descriptor protocol so validators can be attached to classes and will
    manage per-instance storage.
    """

    def __set_name__(self, owner, name):
        """Set the mangled storage name for the metadata."""
        self.storage_name = f"__cbl_metadata_{name}"

    def __set__(self, instance, value):
        """Set the metadata value on the instance after validation."""
        value = self.validate(self.storage_name, value)
        instance.__dict__[self.storage_name] = value

    def __get__(self, instance, owner):
        """Return the stored metadata value for `instance` or the descriptor.

        When accessed on the class, the descriptor object is returned so the
        validator can be inspected.
        """
        if instance is None:
            return self
        return instance.__dict__[self.storage_name]

    @abc.abstractmethod
    def validate(self, name, value):
        """Validate and return `value` or raise an exception on failure."""
        ...

    def __repr__(self):
        """Return a string representation of the validated metadata."""
        return f"{self.__class__.__name__}(value={self.storage_name})"


class Color(Validated):
    """Descriptor that validates color values using PyVista.

    Accepts any input that `pyvista.Color` recognizes (named colors, RGB
    tuples, hex strings, etc.).
    """

    def __init__(self):
        """Initialize the color validator descriptor."""
        super().__init__()

    def validate(self, name, value):
        """Validate that `value` can be interpreted as a color by PyVista.

        Raises:
            ValueError: If `value` cannot be converted to a color.
        
        """
        try:
            pyvista.Color(value)
        except Exception as e:
            raise ValueError(f"{name} must be a valid color value: {e}")

    def __repr__(self):
        """Return a representation for the color validator."""
        return super().__repr__()


class NonBlank(Validated):
    """Descriptor that ensures a string value is not blank after stripping.

    Useful for identifiers or short text fields that must not be empty.
    """

    def __init__(self):
        """Initialize the NonBlank validator descriptor."""
        super().__init__()

    def validate(self, name, value):
        """Validate the subject identifier.

        Strips surrounding whitespace and ensures the resulting string is
        non-empty.
        """
        value = value.strip()
        if not value:
            raise ValueError(f"{name} cannot be blank.")
        return value


class Timepoint(Validated):
    """Metadata for a timepoint in the Kneemesher project."""

    def validate(self, name, value):
        """Validate the timepoint metadata."""
        match value:
            case int() | float():
                if value < 0:
                    raise ValueError(f"{name} must be a non-negative number.")
                return int(value)
            case datetime():
                return value.timestamp()
            case _:
                raise TypeError(
                    f"{name} must be an integer or datetime object, not {type(value)}."
                )


class RegexMatched(NonBlank, Validated):
    """Metadata that is validated against a regular expression."""

    def __init__(self, pattern: str, flags=0):
        """Initialize with a regex pattern and optional flags.

        Args:
            pattern (str): The regular expression pattern to use for validation.
            flags (int, optional): Regex flags (default: 0).
        
        """
        super().__init__()
        self.regex = re.compile(pattern, flags)

    def validate(self, name, value):
        """Validate the value against the regex pattern."""
        value = super().validate(name, value)
        if not self.regex.match(value):
            raise ValueError(
                f"{name} does not match the required pattern: {self.regex.pattern}"
            )
        return value


class ValidEnum(Validated):
    """Metadata that is validated against an Enum."""

    def __init__(self, enum_type: Enum):
        """Initialize with an Enum type."""
        super().__init__()
        self.enum_type = enum_type

    def validate(self, name, value):
        """Validate the value against the Enum."""
        if isinstance(value, self.enum_type):
            return value
        elif isinstance(value, str):
            try:
                return self.enum_type(value)
            except ValueError:
                raise ValueError(f"{name} must be a valid {self.enum_type.__name__}.")
        else:
            raise TypeError(
                f"{name} must be a {self.enum_type.__name__} or string, not {type(value)}."
            )


class ValidVersionString(Validated):
    """Metadata that is validated as a version string."""

    def __init__(self):
        """Initialize the version string validator."""
        super().__init__()

    def validate(self, name, value):
        """Validate the value as a version string."""
        from packaging.version import Version, InvalidVersion

        try:
            Version(value)
            return value
        except InvalidVersion:
            raise ValueError(
                f"{name} must be a PEP 440 valid version string (e.g., '1.0.0')."
            )
