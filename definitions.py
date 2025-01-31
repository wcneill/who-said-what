import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.absolute()

if __name__ == '__main__':
    print(ROOT_DIR)
    print(os.path.join(ROOT_DIR, 'test.txt'))