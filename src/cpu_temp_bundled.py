"""
Hardware Monitor - Optimized for Speed
=======================================
Fast sensor reading with caching and single-pass data collection.

Usage:
    from cpu_temp_bundled import HardwareMonitor
    
    with HardwareMonitor() as monitor:
        data = monitor.get_all_data_flat()
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")

# Sensor type mapping (precomputed)
_SENSOR_TYPE_MAP = {
    "Temperature": "temp",
    "Clock": "clock",
    "Load": "load",
    "Power": "power",
    "Voltage": "volt",
    "Fan": "fan",
    "Control": "ctrl",
    "SmallData": "mem",
    "Data": "data",
    "Throughput": "thru"
}

# Hardware type prefixes (precomputed)
_HW_TYPE_PREFIX = {
    "Cpu": "cpu",
    "GpuNvidia": "gpu",
    "GpuAmd": "gpu",
    "GpuIntel": "gpu",
    "Motherboard": "mb",
    "Memory": "ram",
    "Storage": "disk"
}

# Preset: Essential features for anomaly detection
ESSENTIAL_FEATURES = [
    'cpu_temp_core_tctl_tdie',
    'cpu_load_cpu_total',
    'cpu_power_package',
    'gpu_temp_gpu_core',
    'mb_nuvoton_nct6779d_fan_fan_2',
    'ram_load_memory',
]


class HardwareMonitor:
    """Optimized hardware monitor - minimal overhead, maximum speed."""
    
    __slots__ = ('_computer', '_hardware_list', '_disk_count')
    
    def __init__(self, cpu=True, gpu=True, motherboard=True, storage=True, memory=True):
        self._computer = None
        self._hardware_list = []
        self._disk_count = 0
        self._init(cpu, gpu, motherboard, storage, memory)
    
    def _init(self, cpu, gpu, motherboard, storage, memory):
        import clr
        if LIB_DIR not in sys.path:
            sys.path.insert(0, LIB_DIR)
        
        clr.AddReference(os.path.join(LIB_DIR, "LibreHardwareMonitorLib"))
        from LibreHardwareMonitor.Hardware import Computer
        
        c = Computer()
        c.IsCpuEnabled = cpu
        c.IsGpuEnabled = gpu
        c.IsMotherboardEnabled = motherboard
        c.IsStorageEnabled = storage
        c.IsMemoryEnabled = memory
        c.Open()
        
        self._computer = c
        
        # Pre-cache hardware list with their prefixes
        disk_idx = 0
        for hw in c.Hardware:
            hw_type = str(hw.HardwareType)
            
            prefix = None
            for key, val in _HW_TYPE_PREFIX.items():
                if key in hw_type:
                    prefix = val
                    break
            
            if prefix == "disk":
                prefix = f"disk{disk_idx}"
                disk_idx += 1
            
            if prefix:
                self._hardware_list.append((hw, prefix))
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()
    
    def close(self):
        if self._computer:
            self._computer.Close()
            self._computer = None
            self._hardware_list.clear()
    
    def _format_key(self, prefix, stype, name):
        """Format a sensor key consistently."""
        s_name = str(name).lower().replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('/', '_')
        return f"{prefix}_{_SENSOR_TYPE_MAP.get(stype, stype)}_{s_name}"

    def get_all_data_flat(self, keys: list = None) -> dict:
        """
        Get all sensor data.
        INCLUDES standardized keys (cpu_temp, cpu_load, etc) for consistency.
        """
        data = {}
        
        # Trackers for best standard sensors
        cpu_temp_val = None
        cpu_temp_priority = 0 # 1=General, 2=Package/Tctl
        
        gpu_temp_val = None
        gpu_temp_priority = 0
        
        for hw, prefix in self._hardware_list:
            hw.Update()
            
            # Direct sensors
            for sensor in hw.Sensors:
                stype = str(sensor.SensorType)
                if stype in _SENSOR_TYPE_MAP:
                    val = sensor.Value
                    if val is not None:
                        f_val = float(val)
                        
                        # Add raw key
                        # e.g. cpu_temp_core_tctl_tdie
                        raw_key = self._format_key(prefix, stype, str(sensor.Name))
                        if keys is None or raw_key in keys:
                            data[raw_key] = f_val
                            
                        # Logic for standardized keys (Matching get_essential_fast)
                        name = str(sensor.Name)
                        
                        # CPU Temp
                        if prefix == "cpu" and stype == "Temperature":
                            prio = 1
                            if "Tctl" in name or "Package" in name:
                                prio = 2
                            
                            if prio > cpu_temp_priority:
                                cpu_temp_val = f_val
                                cpu_temp_priority = prio
                                
                        # CPU Load
                        if prefix == "cpu" and stype == "Load" and "Total" in name:
                            data['cpu_load'] = f_val
                            
                        # CPU Power
                        if prefix == "cpu" and stype == "Power" and "Package" in name:
                            data['cpu_power'] = f_val
                            
                        # GPU Temp
                        if prefix == "gpu" and stype == "Temperature":
                            prio = 1
                            if "Core" in name:
                                prio = 2
                            if prio > gpu_temp_priority:
                                gpu_temp_val = f_val
                                gpu_temp_priority = prio
            
            # Sub-hardware (Motherboard fans, RAM)
            for sub in hw.SubHardware:
                sub.Update()
                prefix_sub = f"{prefix}_{str(sub.Name).lower().replace(' ', '_')}"
                
                for sensor in sub.Sensors:
                    stype = str(sensor.SensorType)
                    if stype in _SENSOR_TYPE_MAP:
                        val = sensor.Value
                        if val is not None:
                            f_val = float(val)
                             # Raw key
                            raw_key = self._format_key(prefix, stype, f"{sub.Name}_{sensor.Name}")
                            # Simplify MB keys just in case
                            if prefix == "mb": 
                                raw_key = self._format_key(prefix, stype, f"{sensor.Name}")

                            if keys is None or raw_key in keys:
                                data[raw_key] = f_val
                            
                            # Standardized Fan
                            if prefix == "mb" and stype == "Fan" and f_val > 0 and 'fan_rpm' not in data:
                                data['fan_rpm'] = f_val
        
            # RAM Load check
            if prefix == "ram":
                for sensor in hw.Sensors:
                     if str(sensor.SensorType) == "Load" and "Memory" in str(sensor.Name) and sensor.Value is not None:
                         data['ram_load'] = float(sensor.Value)

        # Apply best found temps
        if cpu_temp_val is not None:
            data['cpu_temp'] = cpu_temp_val
        if gpu_temp_val is not None:
            data['gpu_temp'] = gpu_temp_val
        
        return data
    
    def get_essential_fast(self) -> dict:
        """
        Ultra-fast: Only reads essential sensors for thermal anomaly detection.
        Standardized names ensure consistency across different hardware (Intel/AMD/NVIDIA/AMD).

        Returns:
            dict with standardized keys for anomaly detection:
            - cpu_temp: Target variable
            - cpu_load: Workload intensity
            - cpu_power: Heat generation (best indicator)
            - cpu_clock: Performance state / boost status
            - cpu_volt: Voltage (affects power/heat)
            - gpu_temp: GPU thermal state
            - gpu_load: GPU workload
            - gpu_power: GPU heat contribution
            - mb_temp: Motherboard/ambient temperature baseline
            - ram_load: Memory usage
            - fan_rpm: Cooling system response (NOT cheating for anomaly detection)
        """
        data = {
            'cpu_temp': 0.0,
            'cpu_load': 0.0,
            'cpu_power': 0.0,
            'cpu_clock': 0.0,
            'cpu_volt': 0.0,
            'gpu_temp': 0.0,
            'gpu_load': 0.0,
            'gpu_power': 0.0,
            'mb_temp': 0.0,
            'ram_load': 0.0
        }
        
        # Priority search terms for consistency across different hardware
        cpu_temp_priority = ["Tctl", "Package", "Core Max", "Total"]
        gpu_temp_priority = ["Core", "Hot Spot", "Composite"]
        
        found_sensors = set()
        
        for hw, prefix in self._hardware_list:
            if prefix == "cpu":
                hw.Update()
                for s in hw.Sensors:
                    stype = str(s.SensorType)
                    name = str(s.Name)
                    val = s.Value

                    if val is None:
                        continue

                    # Standardize CPU Temp
                    if stype == "Temperature":
                        if 'cpu_temp' not in found_sensors:
                            # Try to match priority list
                            for term in cpu_temp_priority:
                                if term in name:
                                    data['cpu_temp'] = float(val)
                                    if term == "Tctl" or term == "Package": # High confidence
                                        found_sensors.add('cpu_temp')
                                    break
                            # Fallback if nothing set yet
                            if data['cpu_temp'] == 0.0:
                                data['cpu_temp'] = float(val)

                    # Standardize CPU Load
                    elif stype == "Load" and "Total" in name:
                        data['cpu_load'] = float(val)

                    # Standardize CPU Power
                    elif stype == "Power" and "Package" in name:
                        data['cpu_power'] = float(val)

                    # NEW: CPU Clock (average or max)
                    elif stype == "Clock":
                        if 'cpu_clock' not in found_sensors:
                            if "Average" in name or "Total" in name:
                                data['cpu_clock'] = float(val)
                                found_sensors.add('cpu_clock')
                            elif data['cpu_clock'] == 0.0:  # Fallback to first clock
                                data['cpu_clock'] = float(val)

                    # NEW: CPU Voltage (VID or Core)
                    elif stype == "Voltage":
                        if 'cpu_volt' not in found_sensors:
                            if "VID" in name or "Core" in name:
                                data['cpu_volt'] = float(val)
                                found_sensors.add('cpu_volt')
                            elif data['cpu_volt'] == 0.0:
                                data['cpu_volt'] = float(val)
            
            elif prefix == "gpu":
                hw.Update()
                for s in hw.Sensors:
                    stype = str(s.SensorType)
                    name = str(s.Name)
                    val = s.Value

                    if val is None:
                        continue

                    # GPU Temperature
                    if stype == "Temperature":
                        # Prefer "Core" temp
                        if "Core" in name:
                            data['gpu_temp'] = float(val)
                            found_sensors.add('gpu_temp')
                        elif 'gpu_temp' not in found_sensors:
                            data['gpu_temp'] = float(val)

                    # NEW: GPU Load
                    elif stype == "Load":
                        if 'gpu_load' not in found_sensors:
                            if "Core" in name or "GPU" in name:
                                data['gpu_load'] = float(val)
                                found_sensors.add('gpu_load')
                            elif data['gpu_load'] == 0.0:
                                data['gpu_load'] = float(val)

                    # NEW: GPU Power
                    elif stype == "Power":
                        if 'gpu_power' not in found_sensors:
                            if "Total" in name or "Package" in name:
                                data['gpu_power'] = float(val)
                                found_sensors.add('gpu_power')
                            elif data['gpu_power'] == 0.0:
                                data['gpu_power'] = float(val)
            
            elif prefix == "mb":
                hw.Update()

                # Check main motherboard sensors
                for s in hw.Sensors:
                    stype = str(s.SensorType)
                    name = str(s.Name)
                    val = s.Value

                    if val is None:
                        continue

                    # Motherboard temperature (chipset, ambient)
                    if stype == "Temperature" and 'mb_temp' not in found_sensors:
                        if "System" in name or "Motherboard" in name or "Chipset" in name:
                            data['mb_temp'] = float(val)
                            found_sensors.add('mb_temp')
                        elif data['mb_temp'] == 0.0:
                            data['mb_temp'] = float(val)

                    # Fan RPM
                    elif stype == "Fan" and 'fan_rpm' not in data and val > 0:
                        data['fan_rpm'] = float(val)

                # Check sub-hardware (fan controllers, etc.)
                for sub in hw.SubHardware:
                    sub.Update()
                    for s in sub.Sensors:
                        stype = str(s.SensorType)
                        val = s.Value

                        if val is None:
                            continue

                        # Fan RPM from sub-hardware
                        if stype == "Fan" and 'fan_rpm' not in data and val > 0:
                            data['fan_rpm'] = float(s.Value)
                            break
            
            elif prefix == "ram":
                hw.Update()
                for s in hw.Sensors:
                    if str(s.SensorType) == "Load" and "Memory" in str(s.Name):
                        if s.Value is not None:
                            data['ram_load'] = float(s.Value)
                        break
        
        return data
    
    def get_cpu_data_flat(self) -> dict:
        """Get only CPU data (faster if you only need CPU)."""
        data = {}
        for hw, prefix in self._hardware_list:
            if prefix == "cpu":
                hw.Update()
                for sensor in hw.Sensors:
                    stype = str(sensor.SensorType)
                    if stype in _SENSOR_TYPE_MAP:
                        val = sensor.Value
                        if val is not None:
                            name = str(sensor.Name).lower().replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('/', '_')
                            data[f"cpu_{_SENSOR_TYPE_MAP[stype]}_{name}"] = float(val)
                break
        return data
    
    def get_gpu_data_flat(self) -> dict:
        """Get only GPU data."""
        data = {}
        for hw, prefix in self._hardware_list:
            if prefix == "gpu":
                hw.Update()
                for sensor in hw.Sensors:
                    stype = str(sensor.SensorType)
                    if stype in _SENSOR_TYPE_MAP:
                        val = sensor.Value
                        if val is not None:
                            name = str(sensor.Name).lower().replace(' ', '_').replace('#', '').replace('(', '').replace(')', '').replace('/', '_')
                            data[f"gpu_{_SENSOR_TYPE_MAP[stype]}_{name}"] = float(val)
                break
        return data

    def get_hardware_info(self) -> dict:
        """
        Get hardware names and types.

        Returns:
            dict with hardware information:
            - cpu: {'name': str, 'type': str}
            - gpu: {'name': str, 'type': str}
            - motherboard: {'name': str, 'type': str}
            - ram: {'name': str, 'type': str}
            - storage: [{'name': str, 'type': str}, ...]
        """
        info = {
            'cpu': None,
            'gpu': None,
            'motherboard': None,
            'ram': None,
            'storage': []
        }

        for hw, prefix in self._hardware_list:
            hw_name = str(hw.Name)
            hw_type = str(hw.HardwareType)

            if prefix == "cpu":
                info['cpu'] = {'name': hw_name, 'type': hw_type}
            elif prefix == "gpu":
                info['gpu'] = {'name': hw_name, 'type': hw_type}
            elif prefix == "mb":
                info['motherboard'] = {'name': hw_name, 'type': hw_type}
            elif prefix == "ram":
                info['ram'] = {'name': hw_name, 'type': hw_type}
            elif prefix.startswith("disk"):
                info['storage'].append({'name': hw_name, 'type': hw_type})

        return info

    # Ultra-fast single value properties
    @property
    def cpu_temp(self) -> float | None:
        for hw, prefix in self._hardware_list:
            if prefix == "cpu":
                hw.Update()
                for s in hw.Sensors:
                    if str(s.SensorType) == "Temperature" and s.Value is not None:
                        return float(s.Value)
        return None
    
    @property
    def gpu_temp(self) -> float | None:
        for hw, prefix in self._hardware_list:
            if prefix == "gpu":
                hw.Update()
                for s in hw.Sensors:
                    if str(s.SensorType) == "Temperature" and s.Value is not None:
                        return float(s.Value)
        return None
    
    @property
    def cpu_load(self) -> float | None:
        for hw, prefix in self._hardware_list:
            if prefix == "cpu":
                hw.Update()
                for s in hw.Sensors:
                    if str(s.SensorType) == "Load" and "Total" in str(s.Name):
                        return float(s.Value) if s.Value else None
        return None
    
    @property
    def cpu_power(self) -> float | None:
        for hw, prefix in self._hardware_list:
            if prefix == "cpu":
                hw.Update()
                for s in hw.Sensors:
                    if str(s.SensorType) == "Power" and "Package" in str(s.Name):
                        return float(s.Value) if s.Value else None
        return None


# Benchmark test
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    with HardwareMonitor() as monitor:
        # Warmup
        monitor.get_all_data_flat()
        
        # Benchmark get_all_data_flat
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            data = monitor.get_all_data_flat()
        elapsed = time.perf_counter() - start
        
        print(f"\nget_all_data_flat():")
        print(f"  {iterations} iterations in {elapsed:.3f}s")
        print(f"  {elapsed/iterations*1000:.2f}ms per call")
        print(f"  {iterations/elapsed:.1f} calls/second")
        print(f"  {len(data)} features per call")
        
        # Benchmark cpu_temp property
        start = time.perf_counter()
        for _ in range(iterations):
            t = monitor.cpu_temp
        elapsed = time.perf_counter() - start
        
        print(f"\ncpu_temp property:")
        print(f"  {iterations} iterations in {elapsed:.3f}s")
        print(f"  {elapsed/iterations*1000:.2f}ms per call")
        
        # Show sample data
        print(f"\n" + "=" * 60)
        print("SAMPLE DATA:")
        print("=" * 60)
        for k, v in list(data.items())[:10]:
            print(f"  {k}: {v}")
        print(f"  ... ({len(data)} total features)")
