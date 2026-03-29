from screeninfo import get_monitors  # type: ignore[reportMissingImports]


class MonitorMapper:
    def __init__(self):
        self.monitors = sorted(get_monitors(), key=lambda monitor: monitor.x)
        if not self.monitors:
            raise RuntimeError("No monitors detected")
        self.primary_monitor = self._detect_primary_monitor()
        self._print_layout()

    def _detect_primary_monitor(self):
        for monitor in self.monitors:
            if getattr(monitor, "is_primary", False):
                return monitor

        # Common fallback: primary display has x == 0 on Windows.
        for monitor in self.monitors:
            if monitor.x == 0:
                return monitor

        # Last fallback to avoid blocking startup on unusual layouts.
        return self.monitors[0]

    def _print_layout(self):
        print("Detected monitor layout:")
        for idx, monitor in enumerate(self.monitors, start=1):
            primary_label = " [PRIMARY]" if monitor is self.primary_monitor else ""
            print(
                f"  Screen {idx}: {monitor.width}x{monitor.height} "
                f"at offset ({monitor.x}, {monitor.y}){primary_label}"
            )

    def get_active_monitor(self, cursor_x):
        for monitor in self.monitors:
            if monitor.x <= cursor_x < monitor.x + monitor.width:
                return monitor
        return self.monitors[0]

    def vector_to_screen_coords(self, norm_x, norm_y, monitor_index=None):
        """
        Convert normalized coordinates in [-1, 1] to absolute desktop coordinates.
        """
        norm_x = max(-1.0, min(1.0, norm_x))
        norm_y = max(-1.0, min(1.0, norm_y))

        if monitor_index is None:
            min_x = min(monitor.x for monitor in self.monitors)
            max_x = max(monitor.x + monitor.width for monitor in self.monitors)
            min_y = min(monitor.y for monitor in self.monitors)
            max_y = max(monitor.y + monitor.height for monitor in self.monitors)

            total_h = max_y - min_y

            # Keep neutral gaze centered on the primary monitor center while
            # still spanning the full virtual desktop (supports left-side secondaries).
            primary_center_x = self.primary_monitor.x + self.primary_monitor.width // 2
            if norm_x < 0:
                left_span = max(primary_center_x - min_x, 1)
                px = int(primary_center_x + norm_x * left_span)
            else:
                right_span = max(max_x - primary_center_x, 1)
                px = int(primary_center_x + norm_x * right_span)

            py = int(min_y + (norm_y + 1.0) * 0.5 * total_h)
            return px, py

        monitor = self.monitors[monitor_index]
        px = monitor.x + int((norm_x + 1.0) * 0.5 * monitor.width)
        py = monitor.y + int((norm_y + 1.0) * 0.5 * monitor.height)
        return px, py
