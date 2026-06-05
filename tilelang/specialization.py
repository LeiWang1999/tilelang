"""Shape-guided variant specialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import inspect
import operator
from typing import Any


class SpecializationError(ValueError):
    """Raised when a specialization spec is invalid or cannot be evaluated."""


def _coerce_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _stable_value(value: Any) -> Any:
    value = _coerce_scalar(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return tuple(_stable_value(item) for item in value)
    raise SpecializationError(
        f"Specialization values must be stable scalars or tuples of stable values, got {type(value).__name__}"
    )


def _looks_like_runtime_tensor(value: Any) -> bool:
    return hasattr(value, "shape") or hasattr(value, "stride") or hasattr(value, "strides")


def _cache_value(value: Any) -> Any:
    value = _coerce_scalar(value)
    if _looks_like_runtime_tensor(value):
        return None
    try:
        return _stable_value(value)
    except SpecializationError:
        return repr(value)


class MetadataExpr:
    """Base class for runtime metadata expressions."""

    def evaluate(self, arguments: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def referenced_args(self) -> set[str]:
        raise NotImplementedError

    def _compare(self, other: Any, op_name: str) -> ComparisonExpr:
        return ComparisonExpr(op_name, self, _ensure_expr(other))

    def __eq__(self, other: Any) -> ComparisonExpr:  # type: ignore[override]
        return self._compare(other, "eq")

    def __ne__(self, other: Any) -> ComparisonExpr:  # type: ignore[override]
        return self._compare(other, "ne")

    def __lt__(self, other: Any) -> ComparisonExpr:
        return self._compare(other, "lt")

    def __le__(self, other: Any) -> ComparisonExpr:
        return self._compare(other, "le")

    def __gt__(self, other: Any) -> ComparisonExpr:
        return self._compare(other, "gt")

    def __ge__(self, other: Any) -> ComparisonExpr:
        return self._compare(other, "ge")

    def __bool__(self) -> bool:
        raise TypeError("TileLang metadata expressions cannot be used as Python booleans")


@dataclass(frozen=True, eq=False)
class LiteralExpr(MetadataExpr):
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _stable_value(self.value))

    def evaluate(self, arguments: Mapping[str, Any]) -> Any:
        return self.value

    def referenced_args(self) -> set[str]:
        return set()


@dataclass(frozen=True, eq=False)
class ArgExpr(MetadataExpr):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SpecializationError("tilelang.arg() requires a non-empty argument name")

    @property
    def shape(self) -> ShapeAccessor:
        return ShapeAccessor(self)

    def evaluate(self, arguments: Mapping[str, Any]) -> Any:
        if self.name not in arguments:
            raise SpecializationError(f"Missing runtime argument for specialization: {self.name}")
        return arguments[self.name]

    def referenced_args(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class ShapeAccessor:
    source: ArgExpr

    def __getitem__(self, index: int) -> ShapeExpr:
        if not isinstance(index, int) or index < 0:
            raise SpecializationError("Shape metadata indexes must be non-negative integers")
        return ShapeExpr(self.source, index)


@dataclass(frozen=True, eq=False)
class ShapeExpr(MetadataExpr):
    source: ArgExpr
    index: int

    def evaluate(self, arguments: Mapping[str, Any]) -> Any:
        value = self.source.evaluate(arguments)
        if not hasattr(value, "shape"):
            raise SpecializationError(f"Runtime argument `{self.source.name}` has no shape metadata")
        shape = value.shape
        try:
            size = shape[self.index]
        except IndexError as exc:
            raise SpecializationError(
                f"Runtime argument `{self.source.name}` has no shape dimension {self.index}"
            ) from exc
        except TypeError as exc:
            raise SpecializationError(
                f"Runtime argument `{self.source.name}` shape does not support indexing"
            ) from exc
        return _stable_value(size)

    def referenced_args(self) -> set[str]:
        return self.source.referenced_args()


@dataclass(frozen=True, eq=False)
class ComparisonExpr(MetadataExpr):
    op_name: str
    lhs: MetadataExpr
    rhs: MetadataExpr

    _OPS: dict[str, Callable[[Any, Any], bool]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._OPS is None:
            object.__setattr__(
                self,
                "_OPS",
                {
                    "eq": operator.eq,
                    "ne": operator.ne,
                    "lt": operator.lt,
                    "le": operator.le,
                    "gt": operator.gt,
                    "ge": operator.ge,
                },
            )
        if self.op_name not in self._OPS:
            raise SpecializationError(f"Unsupported requirement operator: {self.op_name}")

    def evaluate(self, arguments: Mapping[str, Any]) -> bool:
        return bool(self._OPS[self.op_name](self.lhs.evaluate(arguments), self.rhs.evaluate(arguments)))

    def referenced_args(self) -> set[str]:
        return self.lhs.referenced_args() | self.rhs.referenced_args()


def _ensure_expr(value: Any) -> MetadataExpr:
    if isinstance(value, MetadataExpr):
        return value
    return LiteralExpr(value)


@dataclass(frozen=True)
class BucketSelector:
    expr: MetadataExpr
    buckets: tuple[int, ...]
    policy: str = "ceil"
    overflow: str = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.expr, MetadataExpr):
            raise SpecializationError("tilelang.bucket() requires a metadata expression")
        if self.policy != "ceil":
            raise SpecializationError(f"Unsupported bucket policy for shape-guided specialization: {self.policy}")
        if self.overflow != "error":
            raise SpecializationError(f"Unsupported bucket overflow behavior for shape-guided specialization: {self.overflow}")
        if not self.buckets:
            raise SpecializationError("tilelang.bucket() requires at least one bucket")
        previous = None
        normalized = []
        for bucket_value in self.buckets:
            bucket_value = _stable_value(bucket_value)
            if not isinstance(bucket_value, int) or isinstance(bucket_value, bool):
                raise SpecializationError("Bucket values must be integers")
            if previous is not None and bucket_value <= previous:
                raise SpecializationError("Bucket values must be strictly increasing")
            normalized.append(bucket_value)
            previous = bucket_value
        object.__setattr__(self, "buckets", tuple(normalized))

    def evaluate(self, arguments: Mapping[str, Any]) -> int:
        value = self.expr.evaluate(arguments)
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpecializationError(f"Bucket selector expected an integer metadata value, got {type(value).__name__}")
        for bucket_value in self.buckets:
            if value <= bucket_value:
                return bucket_value
        raise SpecializationError(
            f"Specialization value {value} exceeds largest bucket {self.buckets[-1]} with overflow='error'"
        )

    def referenced_args(self) -> set[str]:
        return self.expr.referenced_args()


@dataclass(frozen=True)
class SpecializationResult:
    config: tuple[tuple[str, Any], ...]

    @property
    def as_kwargs(self) -> dict[str, Any]:
        return dict(self.config)

    @property
    def cache_key(self) -> tuple[tuple[str, Any], ...]:
        return self.config


@dataclass(frozen=True)
class SpecializationSpec:
    fields: tuple[tuple[str, BucketSelector | Any], ...]
    requirements: tuple[ComparisonExpr, ...]
    compile: str = "on_miss"

    @classmethod
    def create(
        cls,
        fields: Mapping[str, Any],
        *,
        require: Iterable[ComparisonExpr] | None = None,
        compile: str = "on_miss",
    ) -> SpecializationSpec:
        if compile != "on_miss":
            raise SpecializationError(
                f"Unsupported specialization compile mode `{compile}`. "
                "`compile='on_miss'` is the only mode implemented; ahead-of-time dispatch modes are future work."
            )
        normalized_fields = []
        for name, value in fields.items():
            if not isinstance(name, str) or not name:
                raise SpecializationError("Specialization field names must be non-empty strings")
            if isinstance(value, BucketSelector):
                normalized_fields.append((name, value))
            elif isinstance(value, MetadataExpr):
                raise SpecializationError("Metadata expressions must be wrapped in tilelang.bucket() for specialization")
            elif callable(value):
                raise SpecializationError("Callable specialization selectors are not supported in the MVP")
            else:
                normalized_fields.append((name, _stable_value(value)))

        normalized_requirements = []
        for requirement in require or ():
            if not isinstance(requirement, ComparisonExpr):
                raise SpecializationError("Specialization requirements must be metadata comparison expressions")
            normalized_requirements.append(requirement)

        return cls(tuple(normalized_fields), tuple(normalized_requirements), compile)

    @property
    def field_names(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.fields)

    def referenced_args(self) -> frozenset[str]:
        refs: set[str] = set()
        for _, value in self.fields:
            if isinstance(value, BucketSelector):
                refs.update(value.referenced_args())
        for requirement in self.requirements:
            refs.update(requirement.referenced_args())
        return frozenset(refs)

    def bind_arguments(
        self,
        signature: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> inspect.BoundArguments:
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError as exc:
            raise SpecializationError(f"Cannot bind arguments for specialization: {exc}") from exc
        bound.apply_defaults()
        return bound

    def evaluate(self, arguments: Mapping[str, Any]) -> SpecializationResult:
        for requirement in self.requirements:
            if not requirement.evaluate(arguments):
                raise SpecializationError("Specialization requirement failed")

        config = {}
        for name, value in self.fields:
            if isinstance(value, BucketSelector):
                selected = value.evaluate(arguments)
            else:
                selected = value
            config[name] = _stable_value(selected)
        return SpecializationResult(tuple(sorted(config.items())))

    def evaluate_call(
        self,
        signature: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[SpecializationResult, inspect.BoundArguments]:
        bound = self.bind_arguments(signature, args, kwargs)
        return self.evaluate(bound.arguments), bound

    def cache_key_for_call(
        self,
        signature: inspect.Signature,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        result: SpecializationResult,
    ) -> tuple[Any, ...]:
        bound = self.bind_arguments(signature, args, kwargs)
        ignored = self.referenced_args() | self.field_names
        stable_inputs = []
        for name, value in bound.arguments.items():
            if name in ignored:
                continue
            cached = _cache_value(value)
            if cached is not None:
                stable_inputs.append((name, cached))
        return ("specialization", tuple(sorted(stable_inputs)), result.cache_key)


def arg(name: str) -> ArgExpr:
    return ArgExpr(name)


def bucket(
    expr: MetadataExpr,
    buckets: Iterable[int],
    *,
    policy: str = "ceil",
    overflow: str = "error",
) -> BucketSelector:
    return BucketSelector(expr, tuple(buckets), policy=policy, overflow=overflow)


def specialize(
    *,
    require: Iterable[ComparisonExpr] | None = None,
    compile: str = "on_miss",
    **fields: Any,
):
    spec = SpecializationSpec.create(fields, require=require, compile=compile)

    def decorator(func: Any) -> Any:
        func.__tilelang_specialization__ = spec
        if hasattr(func, "specialization"):
            func.specialization = spec
        return func

    return decorator


def get_specialization_spec(func: Any) -> SpecializationSpec | None:
    return getattr(func, "__tilelang_specialization__", None)
