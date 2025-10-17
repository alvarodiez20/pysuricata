"""SVG donut chart renderer for column type visualization."""

import math


class DonutChartRenderer:
    """Renders interactive SVG donut charts with tooltips."""

    def __init__(self):
        self.width = 120
        self.height = 120
        self.cx = 60
        self.cy = 60
        self.outer_radius = 60
        self.inner_radius = 46  # Creates donut ring with styled center hole

    def render_dtype_donut(
        self, numeric: int, categorical: int, datetime: int, boolean: int
    ) -> str:
        """Generate SVG donut chart with tooltips for column types.

        Args:
            numeric: Number of numeric columns
            categorical: Number of categorical columns
            datetime: Number of datetime columns
            boolean: Number of boolean columns

        Returns:
            HTML string containing SVG donut chart with interactive segments
        """
        total = numeric + categorical + datetime + boolean

        if total == 0:
            return self._render_empty_donut()

        # Define segments with colors matching existing CSS
        segments = [
            {"label": "Numeric", "count": numeric, "color": "#4ea3f1"},
            {"label": "Categorical", "count": categorical, "color": "#8ac926"},
            {"label": "Datetime", "count": datetime, "color": "#ffca3a"},
            {"label": "Boolean", "count": boolean, "color": "#ff595e"},
        ]

        return self._build_svg_donut(segments, total)

    def _render_empty_donut(self) -> str:
        """Render an empty donut when no columns exist."""
        return f"""
        <svg class="dtype-donut-svg" viewBox="0 0 {self.width} {self.height}"
             width="{self.width}" height="{self.height}">
            <circle cx="{self.cx}" cy="{self.cy}" r="{self.outer_radius}"
                    fill="#e0e0e0" opacity="0.3"/>
            <circle cx="{self.cx}" cy="{self.cy}" r="{self.inner_radius}"
                    fill="var(--chip-bg-light)" class="donut-hole"/>
            <text x="{self.cx}" y="{self.cy}"
                  text-anchor="middle" dominant-baseline="middle"
                  font-size="12" fill="currentColor" opacity="0.5">
                No data
            </text>
        </svg>
        """

    def _build_svg_donut(self, segments: list[dict], total: int) -> str:
        """Build complete SVG donut with interactive segments.

        Args:
            segments: List of segment dictionaries with label, count, and color
            total: Total number of columns

        Returns:
            SVG HTML string
        """
        # Add background circle to ensure complete appearance even with zero segments
        background_circle = f"""
            <circle cx="{self.cx}" cy="{self.cy}" r="{self.outer_radius}"
                    fill="#f0f0f0" opacity="0.15" class="donut-background"/>"""

        paths = []
        current_angle = -90  # Start at top (12 o'clock position)

        for segment in segments:
            # Calculate percentage and angle for all segments
            pct = (segment["count"] / total) * 100 if total > 0 else 0
            angle_degrees = (segment["count"] / total) * 360 if total > 0 else 0

            # Skip drawing if count is 0, but still track angle
            if segment["count"] == 0:
                current_angle += angle_degrees
                continue

            # Generate arc path
            arc_path = self._calculate_donut_arc_path(
                current_angle, current_angle + angle_degrees
            )

            # Create segment group with data attributes for custom tooltips
            segment_html = f"""
            <g class="donut-segment" data-type="{segment["label"].lower()}">
                <path d="{arc_path}"
                      fill="{segment["color"]}"
                      class="segment-path"
                      data-type="{segment["label"]}"
                      data-count="{segment["count"]}"
                      data-percentage="{pct:.1f}">
                </path>
            </g>"""

            paths.append(segment_html)
            current_angle += angle_degrees

        # Build inner segments with lighter colors (same pattern but faded)
        inner_paths = []
        current_angle = -90

        for segment in segments:
            pct = (segment["count"] / total) * 100 if total > 0 else 0
            angle_degrees = (segment["count"] / total) * 360 if total > 0 else 0

            if segment["count"] == 0:
                current_angle += angle_degrees
                continue

            # Calculate inner ring path (shows lighter version of same colors)
            inner_arc = self._calculate_inner_ring_path(
                current_angle, current_angle + angle_degrees
            )

            inner_segment = f"""
            <path d="{inner_arc}"
                  class="segment-inner"/>"""

            inner_paths.append(inner_segment)
            current_angle += angle_degrees

        # Build complete SVG - outer segments first, then inner on top
        return f"""
        <svg class="dtype-donut-svg" viewBox="0 0 {self.width} {self.height}"
             width="{self.width}" height="{self.height}"
             xmlns="http://www.w3.org/2000/svg">
            {background_circle}
            <g class="donut-segments">
                {"".join(paths)}
            </g>
            <g class="donut-inner-segments">
                {"".join(inner_paths)}
            </g>
        </svg>
        """

    def _calculate_donut_arc_path(self, start_angle: float, end_angle: float) -> str:
        """Calculate SVG path for a full pie slice segment from center to edge.

        Args:
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees

        Returns:
            SVG path string for pie slice
        """
        # Convert degrees to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate arc points on outer edge
        outer_start_x = self.cx + self.outer_radius * math.cos(start_rad)
        outer_start_y = self.cy + self.outer_radius * math.sin(start_rad)
        outer_end_x = self.cx + self.outer_radius * math.cos(end_rad)
        outer_end_y = self.cy + self.outer_radius * math.sin(end_rad)

        # Determine if this is a large arc (> 180 degrees)
        large_arc_flag = 1 if (end_angle - start_angle) > 180 else 0

        # Build the path - FULL PIE SLICE from center to edge
        path = (
            f"M {self.cx},{self.cy} "  # Move to CENTER
            f"L {outer_start_x:.2f},{outer_start_y:.2f} "  # Line to arc start
            f"A {self.outer_radius},{self.outer_radius} 0 {large_arc_flag},1 {outer_end_x:.2f},{outer_end_y:.2f} "  # Arc along edge
            f"Z"  # Close back to center
        )

        return path

    def _calculate_inner_ring_path(self, start_angle: float, end_angle: float) -> str:
        """Calculate SVG path for inner portion (lighter colored pie slice).

        Args:
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees

        Returns:
            SVG path string for smaller inner pie slice
        """
        inner_radius = 45  # Larger radius for more visible inner portion

        # Convert degrees to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate arc points on inner radius
        arc_start_x = self.cx + inner_radius * math.cos(start_rad)
        arc_start_y = self.cy + inner_radius * math.sin(start_rad)
        arc_end_x = self.cx + inner_radius * math.cos(end_rad)
        arc_end_y = self.cy + inner_radius * math.sin(end_rad)

        # Determine if this is a large arc
        large_arc_flag = 1 if (end_angle - start_angle) > 180 else 0

        # Build the path - smaller pie slice from center
        path = (
            f"M {self.cx},{self.cy} "  # Move to CENTER
            f"L {arc_start_x:.2f},{arc_start_y:.2f} "  # Line to arc start
            f"A {inner_radius},{inner_radius} 0 {large_arc_flag},1 {arc_end_x:.2f},{arc_end_y:.2f} "  # Arc
            f"Z"  # Close back to center
        )

        return path
