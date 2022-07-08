# file_deployment_manager

The project solves a problem of the best distribution of files on the servers
in terms of cost/speed of access on the computer network.

The original problem was as follows:

```
There is an abstract computer network. It consists of personal computers,
servers and two layers of switches between them (the first connected to the
computers, the second to the servers, both connected to each other).

The servers store files that need to be shared by each computer. A program must
be created that calculates an acceptable result for storing files on the servers
based on the input data.

Restrictive conditions:
- each server has a storage cost per megabyte
- there is limited (or no) connection speed between devices on neighboring layers
- switches have an internal delay in transferring files
- there is an abstract parameter `coefficient` which defines the ratio between
    the importance of speed and the cost of storage
- files can be stored in a single copy or duplicated on servers, but each
    computer needs access to each file
- each server has a limited storage space
- it is not necessary to calculate `the best' location, only `acceptable'
    (that is, you need to use optimization algorithms in search)

Additional restrictive condition (it was not in the original version, but
logically it should be):
- any file path has the form "server -> cloud switch -> local switch -> computer"
```

Later this program was the basis for the PhD thesis (not mine, I'm only a
bachelor :) ), but that's another story.

This project implements a solution that optimizes the deployment of files
according to specified parameters. Optimization is performed using optimization
algorithms (unexpectedly!), the project implements an annealing simulation
algorithm, genetic algorithm, and ant colony algorithm.

To start the project, you need to load data into the `data.json` file, following
the example of already existing ones. Required data for starting:

```
counts - number of files, computers, first and second level switches and servers
file_sizes - the size of each file in bytes
delays - delays of two switch levels in milliseconds
pc_to_ls - speed matrix between computers and first level switches (in bits)
ls_to_cs - speed matrix between the first and second level switches (in bits)
cs_to_sv - matrix of speeds between the second level switches and servers (in bits)
server_prices - price of storing a megabyte of data on each server
server_spaces - amount of free space on each server
coefficient - abstract number which defines the ratio between the speed and the cost
```

Running is done with the command:

```
python3 main.py [-ant_colony] [-genetic_algorithm] [-simulated_annealing]
```

where you need to specify one or more optimization algorithms. The results are
written to the `result.json` file as a result matrix for each algorithm.
