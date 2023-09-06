import random
import numpy as np
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Converter:
    """
    torch index
    -> numpy action
    numpy action
    -> torch index
    """
    def __init__(self, envname):
        self.envname = envname

    def index2act(self, _input, batch):
        if self.envname == "hope":
            if batch == 1:
                first_action = (_input % 5 / 2) - 1
                sec_action = ((_input % 25 - _input % 5) / 10) - 1
                third_action = ((_input - _input % 25) / 50) - 1
                out = torch.tensor([first_action, sec_action, third_action], device=DEVICE)
            else:
                i = 0
                out = torch.zeros((10, 3), device=DEVICE)
                while i < batch:
                    first_action = (_input[i] % 5 / 2) - 1
                    sec_action = ((_input[i] % 25 - _input[i] % 5) / 10) - 1
                    third_action = ((_input[i] - _input[i] % 25) / 50) - 1
                    out[i] = torch.tensor([first_action, sec_action, third_action], device=DEVICE)
                    i = i + 1
            return out.cpu().numpy()
        elif self.envname == "cart":
            return _input.cpu().numpy()
        elif self.envname == "test":
            action_ary = [[20, 0], [12, 12], [0, 20], [-12, 12], [-20, 0], [-12, -12], [0, -20], [12, -12]]
            action_dict = {0: (5, 0), 1: (3, 3), 2: (0, 5), 3: (-3, 3),
                           4: (-5, 0), 5: (-3, -3), 6: (0, -5), 7: (3, -3)}
            _input = _input.cpu().numpy().astype(np.int64)
            try:
                a = len(_input)
                _input = np.array(_input)
            except:
                _input = np.array([_input])
            values = [action_dict[k] for k in _input if k in action_dict]
            result = [list(item) for item in values]
            return np.squeeze(result)

        else:
            print("converter error")

    def act2index(self, _input, batch):
        if self.envname == "hope":
            if batch == 1:
                _input = _input + 1
                _input = _input * 2
                out = _input[2] * 25 + _input[1] * 5 + _input[0]
            else:
                i = 0
                out = np.zeros(batch)
                while i < batch:
                    _input[i] = _input[i] + 1
                    _input[i] = _input[i] * 2
                    out[i] = _input[i][2] * 25 + _input[i][1] * 5 + _input[i][0]
                    i = i + 1
            return torch.from_numpy(out).to(DEVICE)
        elif self.envname == "cart":
            return torch.from_numpy(_input).to(DEVICE)

        elif self.envname == "test":

            act_keys = [tuple(sublist) for sublist in _input]

            action_dict = {(5, 0): 0, (3, 3): 1, (0, 5): 2, (-3, 3): 3,
                           (-5, 0): 4, (-3, -3): 5, (0, -5): 6, (3, -3): 7}
            values = [action_dict[k] for k in act_keys if k in action_dict]
            values = np.array(values)
            return torch.from_numpy(values).to(DEVICE)

        else:
            print("converter error")

    def rand_act(self):
        if self.envname == "hope":
            return (np.random.randint(5, size=(3,)) - 2)/2
        elif self.envname == "cart":
            _a = np.random.randint(2, size=(1,))
            return _a[0]
        else:
            print("converter error")




