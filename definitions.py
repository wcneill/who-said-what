import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    print(ROOT_DIR)
    print(os.path.join(ROOT_DIR, 'test.txt'))