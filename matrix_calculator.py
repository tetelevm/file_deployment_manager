from __future__ import annotations


class Device:
    """
    A class that describes a single device (i.e. a computer, switch, or
    server). After initialization is immutable.

    Contains the next data:
    * name - name of the device. A string that consists of `2 letters +
        number`.
    * delay - device delay in milliseconds. In fact, only for switches,
        servers and computers do not have them.
    * speeds - dictionary of values {`device name``: ``speed``}.
        Represents all values for the current device to its neighbors.
        Speed in bits per second.
    """

    def __init__(self, name: str, delay: float):
        self.name = name
        self.delay = delay
        self.speeds: dict[str, float] = dict()

    def __str__(self):
        active_connections = [
            name
            for (name, speed) in self.speeds.items()
            if speed > 0
        ]
        return "dev <{name}>, has conn to ({connections})".format(
            name=self.name,
            connections=", ".join(active_connections)
        )


class Path:
    """
    The class of the file path from the server to the computer. It is
    used to calculate the best time in which the user can get the file.

    Contains values:
    * devices - list of devices through which the file passes
    * min_throughput - minimum throughput all the way, its bottle neck
    * delay - total delay of all devices

    The devices data is only needed for as long as the paths are
    calculated, after the calculation the time to get the file is
    sufficient.
    """

    def __init__(self, first_device: Device | list[Device]):
        if type(first_device) == list:
            # for copying
            self.devices = first_device.copy()
            return

        self.devices: list[Device] = [first_device]
        self.min_throughput = 8 * 2 ** 64  # max throughput
        self.delay = first_device.delay  # 0, if first device has no delays

    def add_device(self, device: Device):
        """
        Adds a new device to the list of used devices and updates its
        own parameters.
        """

        self.delay += device.delay
        self.min_throughput = min(
            self.min_throughput,
            self.devices[-1].speeds[device.name]
        )
        self.devices.append(device)

    def copy(self) -> Path:
        """
        Return a copy of the path.
        """

        new_path = Path(self.devices)
        new_path.min_throughput = self.min_throughput
        new_path.delay = self.delay
        return new_path

    def get_transfer_time(self, file_size: float) -> float:
        """
        Calculates how many seconds it takes to transfer a file via a
        this path.
        """

        file_size *= 8  # converting bytes to bits
        push_time = file_size / self.min_throughput  # b / (b/s) = s
        delay_on_sec = self.delay * 1000
        return push_time + delay_on_sec


class ResultMatrix:
    """
    The resulting matrix for finding optimal paths.
    Represents the time each file will take to get from each server to
    each computer. It contains 3 dimensions: [file][server][computer].
    """

    def __init__(self, matrix: list[list[list[float]]]):
        self.matrix = matrix

    def __getitem__(self, index: tuple[int, int, int]) -> float:
        file, server, pc = index
        return self.matrix[file][server][pc]


class Calculator:
    """
    A class for calculating the best path matrix.
    It finds the best paths for each file from each server to each
    computer.

    It needs it to start the calculation:

    * file_sizes - file sizes (in bytes), which are planned to be
        transferred
    * counts - the number of servers, each layer of switches and
        computers
    * delays - delays for each device (in milliseconds)
    * pc_to_ls - speed from computers to local switches (here and below
        in bps)
    * ls_to_cs - speed between local and cloud switches
    * cs_to_sv - speed between the cloud switches and the server

    An example of usage and data can be found in the `main` function.
    """

    def __init__(
        self,
        counts: dict[str, int],
        file_sizes: list[float],
        delays: list[list[float]],
        pc_to_ls: list[list[float]],
        ls_to_cs: list[list[float]],
        cs_to_sv: list[list[float]],
    ):
        self.counts = counts
        self.file_sizes = file_sizes
        if len(delays) == 2:
            # if specified only for switches
            delays = [
                [0] * counts["pc"],  # computer delays
                delays[0],  # local switch delays
                delays[1],  # cloud switch delays
                [0] * counts["sv"]  # server delays
            ]

        # Creating all devices
        # the `self.devices` dictionary has as key the string
        # "{type from [pc, ls, cs, sv]}{number}", for example "ls12", "sv4".
        self.devices: dict[str, Device] = dict()
        for (type_index, dev_type) in enumerate(["pc", "ls", "cs", "sv"]):
            current_delays = delays[type_index]
            for dev_ind in range(self.counts[dev_type]):
                dev_name = f"{dev_type}{dev_ind}"
                self.devices[dev_name] = Device(dev_name, current_delays[dev_ind])

        # Initialize three levels of device connection
        levels = {
            ("pc", "ls"): pc_to_ls,
            ("ls", "cs"): ls_to_cs,
            ("cs", "sv"): cs_to_sv,
        }
        for ((from_, to), array) in levels.items():
            for f_index in range(self.counts[from_]):
                f_device = self.devices[f"{from_}{f_index}"]
                for t_index in range(self.counts[to]):
                    speed = array[f_index][t_index]
                    if speed == 0:
                        continue
                    f_device.speeds[f"{to}{t_index}"] = speed

    def find_best_time(self, file_size: float, sv_name: str, pc_name: str) -> float:
        pass

    def calculate(self) -> ResultMatrix:
        # array of best times from 3 dimensions: [file][server][computer]
        result_matrix = [
            # each file
            [
                # each server
                [
                    # each computer
                    self.find_best_time(
                        self.file_sizes[file_index],
                        f"sv{sv_index}",
                        f"pc{pc_index}",
                    )
                    for pc_index in range(self.counts["pc"])
                ]
                for sv_index in range(self.counts["sv"])
            ]
            for file_index in range(self.counts["file"])
        ]

        return ResultMatrix(result_matrix)


def main():
    # the number of computer, local switch, cloud switch, server
    counts = {"files": 16,  "pc": 3, "ls": 3, "cs": 6, "sv": 2}

    # the size of each file
    sizes = [
        7602176, 6815744, 7077888, 2359296, 6553600, 4980736, 4194304, 1835008,
        4456448, 2359296, 2359296, 1572864, 2883584, 5767168, 7864320, 2621440,
    ]

    # delays for switches (for computers and servers are hidden)
    delays = [
        [100, 75, 100],  # local
        [50, 25, 40, 75, 25, 50],  # cloud
    ]

    pc_to_ls = [
        [231, 0, 0],  # 3 pk
        [0, 482, 0],
        [0, 0, 774],
        # 3 ls
    ]
    ls_to_cs = [
        [0, 115, 113, 809, 0, 0],  # 3 ls
        [727, 0, 0, 0, 108, 0],
        [0, 343, 0, 919, 0, 796],
        # 6 cs
    ]
    cs_to_sv = [
        [507, 0],  # 6 cs
        [120, 666],
        [636, 813],
        [0, 811],
        [0, 972],
        [0, 785],
        # 2 sv
    ]

    calc = Calculator(counts, sizes, delays, pc_to_ls, ls_to_cs, cs_to_sv)
    matrix = calc.calculate()
    print(matrix)


if __name__ == '__main__':
    main()
