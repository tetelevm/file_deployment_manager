class Device:
    """
    A class that describes a single device (i.e. a computer, switch, or
    server). Contains the next data:

    * name - name of the device. A string that consists of `2 letters +
        number`.
    * delay - device delay in milliseconds. In fact, only for switches,
        servers and computers do not have them.
    * speeds - dictionary of values {`device name``: ``speed``}.
        Represents all values for the current device to its neighbors.
        Speed in bits per second.
    """

    def __init__(self, name: str, delay: int):
        self.name: str = name
        self.delay: int = delay
        self.speeds: dict[str, int] = dict()

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

    * file_sizes - file sizes (in mb), which are planned to be
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
        self.sizes = file_sizes
        if len(delays) == 2:
            # if specified only for switches
            delays = [
                [0] * counts["pc"],  # computer delays
                delays[0],  # local switch delays
                delays[1],  # cloud switch delays
                [0] * counts["sv"]  # server delays
            ]
        self.delays = delays

        self.devices: dict[str, Device] = dict()
        self.init_devices(pc_to_ls, ls_to_cs, cs_to_sv)

    def init_devices(self, pc_to_ls, ls_to_cs, cs_to_sv):
        """
        A method for creating devices and connections between them.

        The `self.devices` dictionary has as key the string
        "{type from [pc, ls, cs, sv]}{number}", for example "ls12", "sv4".
        """

        # creating all devices
        for (type_index, dev_type) in enumerate(["pc", "ls", "cs", "sv"]):
            current_delays = self.delays[type_index]
            for dev_ind in range(self.counts[dev_type]):
                dev_name = f"{dev_type}{dev_ind}"
                self.devices[dev_name] = Device(dev_name, current_delays[dev_ind])

        # initialize three levels of device connection
        level_data = [
            {"from": "pc", "to": "ls", "array": pc_to_ls},
            {"from": "ls", "to": "cs", "array": ls_to_cs},
            {"from": "cs", "to": "sv", "array": cs_to_sv},
        ]
        for level in level_data:
            from_, to, array = level["from"], level["to"], level["array"]
            array: list[list[int]]

            for from_index in range(self.counts[from_]):
                from_name = f"{from_}{from_index}"
                for to_index in range(self.counts[to]):
                    to_name = f"{to}{to_index}"
                    speed = array[from_index][to_index]
                    self.devices[from_name].speeds[to_name] = speed
                    self.devices[to_name].speeds[from_name] = speed


def main():
    # the number of computer, local switch, cloud switch, server
    counts = {"files": 20,  "pc": 3, "ls": 3, "cs": 6, "sv": 2}

    # the size of each file
    sizes = [
        25, 92, 93, 64, 71, 56, 14, 37, 65, 22, 61, 95, 100, 99, 24, 13,
        80, 40, 64, 74,
    ]

    # delays for switches (for computers and servers are hidden)
    delays = [
        [4, 6, 2],  # local
        [5, 2, 4, 7, 1, 1],  # cloud
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
    for dev in sorted(calc.devices.values(), key=lambda d: d.name):
        print(dev)


if __name__ == '__main__':
    main()
