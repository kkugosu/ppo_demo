import base64
from varname import nameof
from control import DQN, PG, AC, DDPG, SAC, TRPO, PPO
from utils import render

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    HIDDEN_SIZE = 32
    learning_rate = 0.01
    policy = None

    def getinteger(integer):
        valid = 0
        while valid == 0:
            integer = input("->")
            try:
                int(integer)
                if float(integer).is_integer():
                    valid = 1
                    return int(integer)
                else:
                    print("enter integer")
            except ValueError:
                print("enter integer")


    def getfloat(float_):
        valid = 0
        while valid == 0:
            float_ = input("->")
            try:
                float(float_)
                valid = 1
                return float(float_)
            except ValueError:
                print("enter float")
    envname = None
    control = None

    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        envname = input("->")
        if envname == "cart":
            valid = 1
        elif envname == "hope":
            valid = 1
        elif envname == "test":
            valid = 1
        else:
            print("error")

    valid = 0
    while valid == 0:
        print("enter RL control, {PG, DQN, ...}")
        control = input("->")
        if control == "PG":
            valid = 1
        elif control == "DQN":
            valid = 1
        elif control == "AC":
            valid = 1
        elif control == "TRPO":
            valid = 1
        elif control == "PPO":
            valid = 1
        elif control == "DDPG":
            valid = 1
        elif control == "SAC":
            valid = 1
        else:
            print("error")

    print("enter HIDDEN_SIZE")
    HIDDEN_SIZE = getinteger(HIDDEN_SIZE)

    print("enter batchsize")
    BATCH_SIZE = getinteger(BATCH_SIZE)

    print("enter memory capacity")
    CAPACITY = getinteger(CAPACITY)

    print("train iter will be")
    TRAIN_ITER = getinteger(TRAIN_ITER)

    print("train per memory")
    MEMORY_ITER = getinteger(MEMORY_ITER)

    print("enter learning rate")
    learning_rate = getfloat(learning_rate)

    print("load enter 0 or 1")
    load_ = input("->")
    e_trace = 1

    if control == "PG":
        e_trace = 100
        mechanism = PG.PGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DQN":
        mechanism = DQN.DQNPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DDPG":
        mechanism = DDPG.DDPGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                    learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)

    elif control == "TRPO":
        mechanism = TRPO.TRPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                    learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "PPO":
        mechanism = PPO.PPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "SAC":
        mechanism = SAC.SACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)

    elif control == "AC":
        mechanism = AC.ACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    else:
        print("error")

    my_rend = render.Render(policy, BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate,
                            TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
    my_rend.rend()

