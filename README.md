# file_manager_on_local_network

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
- there is an abstract parameter `coefficient` which defines the correlation
    between the importance of speed and the cost of storage
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
