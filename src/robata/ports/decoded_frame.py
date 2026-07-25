"""Bounded, adapter-neutral decoded grayscale frame views.

The view is deliberately an in-memory observation rather than an artifact format.
Adapters normalize decoded video into a compact row-major 8-bit grayscale raster before
passing it to local visual-quality and adaptive-sampling consumers. Consumers therefore
never have to guess whether arbitrary bytes are encoded media, a PNG/JPEG, or pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

from robata.contracts.common import INT64_MAX, INT64_MIN


@dataclass(frozen=True, slots=True)
class DecodedFrameView:
    """One compact grayscale view with exact timestamp and explicit dimensions.

    ``gray_pixels`` contains exactly ``width * height`` unsigned 8-bit luminance values,
    ordered from left to right within each row and from top to bottom across rows. It has
    no stride or encoded-image header. The source adapter owns any resize/colorspace work
    needed to construct this bounded view.
    """

    timestamp_ns: int
    width: int
    height: int
    gray_pixels: bytes

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer")
        if not INT64_MIN <= self.timestamp_ns <= INT64_MAX:
            raise ValueError("timestamp_ns must be a signed int64")
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.gray_pixels, bytes):
            raise TypeError("gray_pixels must be immutable bytes")
        expected_length = self.width * self.height
        if len(self.gray_pixels) != expected_length:
            raise ValueError(
                "gray_pixels must contain exactly width * height row-major grayscale bytes"
            )

    @property
    def pixel_count(self) -> int:
        """The number of grayscale samples in this compact view."""

        return self.width * self.height


__all__ = ["DecodedFrameView"]
