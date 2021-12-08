# MUZERO

Implementation of the MuZero algorithm based on [this paper](http://arxiv.org/abs/1911.08265) and using [this impmlementation](https://github.com/werner-duvaud/muzero-general) as a reference.

# Usage

To run the environment docker is required. 

```
# Start the docker container with the learning environment, and the redis data store
./env/start -o
# Access tty inside container
./env/terminal
```

To run a training session execute the following command in the appropriate directory inside
the learning environment.

```
python3 ./src/instances/cartpole.py
```


Data will be output to `./results` in the tensorflow event format. 

In the learning environment run the command 

```
tensorboard --bind_all --logdir ./muzero/results
```

and navigate to http://localhost:6006 to view training results. 
