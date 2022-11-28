
import abc

class Config(object):
    def __int__(self,
                seed,
                entropy_const,
                learning_rate,
                num_episodes,
                num_rollouts,
                num_epochs,
                fractional_reward,
                negative_reward,
                observation_spec_shape_x,
                observation_spec_shape_y,
                pretrained,
                pretrained_path,
                checkpoint_path):

        self.seed = seed
        self.entropy_const = entropy_const
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.num_rollouts = num_rollouts
        self.num_epochs = num_epochs
        self.fractional_reward = fractional_reward
        self.negative_reward = negative_reward
        self.observation_spec_shape_x = observation_spec_shape_x
        self.observation_spec_shape_y = observation_spec_shape_y
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.checkpoint_path = checkpoint_path

    @abc.abstractmethod
    def parse_from_file(self, filepath):
        config = Config()

        return config

