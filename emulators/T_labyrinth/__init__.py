from ..environment_creator import BaseEnvironmentCreator


class TLabyrinthCreator(BaseEnvironmentCreator):    
    @staticmethod
    def available_games(**kwargs):
        games = [
          "T_lab"
        ]
        return games

    @staticmethod
    def get_environment_class():
        from .Tlab_emulator import TLabyrinthEmulator
        return TLabyrinthEmulator

    @staticmethod
    def add_required_args(argparser):
        argparser.add_argument('-g', default='T_lab', help='Name of game', dest='game')
        argparser.add_argument('--single_life_episodes', action='store_true',
            help="If True, training episodes will be terminated when a life is lost (for games)",
            dest="single_life_episodes")
        argparser.add_argument('-v', '--visualize', action='store_true',
            help="Show a game window", dest="visualize")
        argparser.add_argument('-rs', '--random_start', default=True, type=bool,
            help="Whether or not to start with 30 noops for each env. Default True",
            dest="random_start")
        argparser.add_argument('--seed', default=3, type=int, help='Sets the random seed.',
            dest='random_seed')
