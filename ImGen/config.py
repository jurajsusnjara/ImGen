import configparser

config = configparser.ConfigParser()


def parse_config(fname):
    config.read(fname)


if __name__ == '__main__':
    parse_config('config.cfg')
    a = config['gan']['g_layers']
