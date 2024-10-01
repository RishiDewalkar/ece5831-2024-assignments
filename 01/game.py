# Importing the module
import draw_1

def play_game():
    return "Winner is Team A"

def main():
    output = play_game()
    draw_1.draw_game(output)

if __name__ == '__main__':
    main()