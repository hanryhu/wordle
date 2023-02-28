import math
import numpy as np
from tqdm import tqdm

'''
Prereqs: Python 3

Download file from https://drive.google.com/file/d/1oGDf1wjWp5RF_X9C7HoedhIWMh5uJs8s/view
$ pip install ansicolor
'''
from colors import green, yellow
import logging
logging.basicConfig()
LOG = logging.getLogger(__file__)
LOG.setLevel(logging.INFO)

def load_words_from_file(filename):
    with open(filename, 'r') as f:
         words = []
         for line in f:
             words.append(line.upper())
    words = [x.strip() for x in words]
    five_letters = [x for x in words if len(x) == 5]
    return five_letters

five_letters = load_words_from_file('wordlewords.txt')
scrabble_five_letters = load_words_from_file('Collins Scrabble Words (2019).txt')
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
GREEN = 0
YELLOW = 1
BLACK = 2

def comments():
    '''
    There are 12972 possible guesses per phase, but there are 3^5 possible clues per phase, so DP is necessary.

    num_guesses(guess, clues, k) = min number of required remaining guesses with k guesses remaining
    num_clues(guess, clues, k) = max number of possible remaining clues with k guesses remaining

    num_guesses(clues, k) =
    '''
    # Theoretically requires >=3 guesses
    # - 1st guess = 1 rational choice
    # - 2nd guess = 3^5 rational choices < 12972 - 1
    # - 3rd guess = 3^5 * 3^5 rational choices > 12972 - 1 - 3^5
    pass

class Clue():
    cache = {}

    def __init__(self, guess, greens, yellows, blacks=None):
        '''
        guess_signature: dict(str : tuple(i, color)): maps each letter
        in the word to a list of doubleton tuples that list the
        character index and the color (enum 0,1,2) of each occurrence
        of that letter in the word.
        '''
        self.guess = guess
        self.greens = list(sorted(greens))
        self.yellows = list(sorted(yellows))
        colors = []
        for i in range(len(self.guess)):
            if i in greens:
                colors.append(GREEN)
            elif i in yellows:
                colors.append(YELLOW)
            else:
                if blacks:
                    assert i in blacks
                colors.append(BLACK)
        self.guess_signature = {
            letter: tuple((i, color) for i, color in enumerate(colors)
                          if guess[i] == letter)
            for letter in set(guess)
        }

    def all_green(self):
        return all(color == GREEN
                   for _, colors in self.guess_signature.items()
                   for i, color in colors)

    def reject(self, word):
        if (self, word) in self.cache:
            LOG.debug(f'reject cached {self} {word}')
            return self.cache[(self, word)]
        word_signature = {
            letter: [i for i in range(len(word))
                     if word[i] == letter]
            for letter in self.guess_signature
        }
        for letter, colors in self.guess_signature.items():
            letter_occurrences = word_signature[letter]
            nonblacks = [i for (i, color) in colors
                         if color != BLACK]
            if len(nonblacks) > len(letter_occurrences):
                # Not enough letter occurrences; extra letter
                # occurrences is allowed unless black letters are
                # present.
                LOG.debug(f'reject 1 because {nonblacks} {letter_occurrences}')
                self.cache[(self, word)] = True
                return True
            if len(colors) - len(nonblacks) and len(nonblacks) != len(letter_occurrences):
                # Black letters restrict letter occurrences.
                LOG.debug(f'reject 2 because {colors} {nonblacks} {letter_occurrences}')
                self.cache[(self, word)] = True
                return True
            for i, color in colors:
                if color == GREEN:
                    if word[i] != self.guess[i]:
                        LOG.debug(f'reject 3 because {word} {self.guess} {i}')
                        self.cache[(self, word)] = True
                        return True
                elif color == YELLOW or color == BLACK:
                    if word[i] == self.guess[i]:
                        LOG.debug(f'reject 4 because {word} {self.guess} {i}')
                        self.cache[(self, word)] = True
                        return True
        self.cache[(self, word)] = False
        return False

    def filter_words(self, words):
        return {word for word in words if not self.reject(word)}

    def get_hashable_rep(self):
        return tuple(sorted(self.guess_signature.items()))

    def __str__(self):
        word_colors = [lambda x: x] * 5
        for letter, colors in self.guess_signature.items():
            for i, color in colors:
                if color == GREEN:
                    word_colors[i] = green
                if color == YELLOW:
                    word_colors[i] = yellow
        return ''.join(word_colors[i](self.guess[i]) for i in range(len(self.guess)))

    def __repr__(self):
        return f'Clue({repr(self.guess)}, {repr(self.greens)}, {repr(self.yellows)})'

    def __hash__(self):
        return hash(self.get_hashable_rep())

    def __eq__(self, other):
        return self.get_hashable_rep() == other.get_hashable_rep()


class ClueCollection():
    # This is tied to a word list bc hashable rep is otherwise hard to implement given clues with duplicate letters possible.
    cache={}

    def __init__(self, word_list=five_letters, clues=[]):
        self.clues = []
        self.word_list = word_list.copy()
        for clue in clues:
            self.add_clue(clue)

    def copy(self):
        other = ClueCollection(self.word_list)
        other.clues = self.clues.copy()
        return other

    def add_clue(self, clue):
        self.clues.append(clue)
        self.word_list = clue.filter_words(self.word_list)

    def reject(self, word):
        cache = self.cache
        if (self, word) in cache:
            return cache[(self, word)]
        if any(clue.reject(word) for clue in self.clues):
            cache[(self, word)] = True
            return True
        cache[(self, word)] = False
        return False

    def filter_words(self, word_list):
        return {word for word in word_list if not self.reject(word)}

    def get_hashable_rep(self):
        return tuple(sorted(self.word_list))

    def __str__(self):
        return '\n'.join(str(clue) for clue in self.clues)

    def __repr__(self):
        return f'ClueCollection({repr(self.word_list)}, {repr(self.clues)})'

    def __hash__(self):
        return hash(self.get_hashable_rep())

    def __eq__(self, other):
        # ???
        return self.get_hashable_rep() == other.get_hashable_rep()


def give_clue(guess, reference):
    assert len(guess) == 5
    letters = set(reference)
    yellows = set()
    greens = set()
    counts = {letter: 0 for letter in letters}
    for letter in reference:
        counts[letter] += 1
    for i in range(len(guess)):
        # Prioritize giving greens over nongreens.
        if guess[i] == reference[i]:
            greens.add(i)
            counts[reference[i]] -= 1
    for i in range(len(guess)):
        # Prioritize giving yellows left to right.
        if i not in greens and counts.get(guess[i], 0) > 0:
            yellows.add(i)
            counts[guess[i]] -= 1
    # Other indices are blacks.
    return Clue(guess, greens, yellows)

print(give_clue('HAPPY', 'HARPY'))
print(give_clue('HARPY', 'HAPPY'))
print(give_clue('ABCDE', 'EDCBA'))
print(give_clue('AAAAA', 'EDCBA')) # Should be 4 blacks and a green.
print(give_clue('AAAAB', 'EDCBA')) # Should be 1 yellow and 3 blacks and a yellow.

ALLOWED_ROUNDS = 1

def optimal_player(clue, clue_collection, guesses_left=ALLOWED_ROUNDS, player_cache={}, adversary_cache={}, nested=False, consider_guesses=scrabble_five_letters):
    '''
    Given guess and word_list, chooses the guess that results in the least number of remaining guesses.
    Args: clue, clue_collection, guesses_left=ALLOWED_ROUNDS, player_cache={}, adversary_cache={}

    Example: optimal_player(None, ClueCollection(five_letters), guesses_left=5)

    Returns the guess that does the best against the adversarial server, and how many more guesses that would take (including this guess).
    '''
    if not clue_collection.word_list:
        # Somehow ruled out all options
        raise Exception(f'illegal word list {clue_collection}\n{clue}')
    if clue and clue.all_green():
        # Guessed the right word last time
        return clue.guess, 0
    if guesses_left == 0:
        # No more time
        return None, math.inf
    if len(clue_collection.word_list) == 1:
        # Only one option left
        for word in clue_collection.word_list:
            return word, 1

    clue_collection = clue_collection.copy()
    if clue:
        clue_collection.add_clue(clue)
    if clue_collection in player_cache:
        best_guess, best_guess_num, cached_guesses_left = player_cache[clue_collection]
        if ((best_guess is None and cached_guesses_left >= guesses_left) # guaranteed be stumped
            or best_guess is not None and best_guess_num <= guesses_left): # guaranteed guess right
            return (best_guess, best_guess_num)
        # Otherwise, recalculate best guess under new guesses_left.

    best_guess = None
    best_guess_num = math.inf

    # Could restrict search space to non-eliminated words (word_list)
    # for an approximation of hard mode, or all words (five_letters) for
    # normal mode.  hard mode should be faster, eliminating lots of options.

    # Or all words from scrabble for hard mode
    guesses = clue_collection.filter_words(consider_guesses) # clue_collection.word_list
    pbar = guesses if nested else tqdm(guesses, leave=False)
    for guess in pbar:
        if not nested:
            pbar.set_description(f"player considering {guess}")
        next_clue, num = adversarial_server(guess, clue_collection, clues_left=guesses_left, player_cache=player_cache, adversary_cache=adversary_cache, nested=True)
        if num < best_guess_num:
            best_guess = guess
            best_guess_num = num
        if num == 1:
            break

    best_guess_num += 1
    # if guesses_left == 1:
    #     print(f'Optimal play for {clue} in:\n'
    #           f'{clue_collection} ({len(clue_collection.word_list)} possibilities):\n'
    #           f'{best_guess} ({best_guess_num} guesses left)')
    if best_guess is not None and guesses_left == 1:
        LOG.debug(f'Found a solution {clue} in:\n{clue_collection}')
    player_cache[clue_collection] = (best_guess, best_guess_num, guesses_left)
    return best_guess, best_guess_num


def adversarial_server(guess, clue_collection, clues_left=ALLOWED_ROUNDS - 1, player_cache={}, adversary_cache={}, nested=False):
    '''
    Given guess and clue_collection, chooses the clue that results
    in the greatest number of remaining guesses.

    Args: guess, clue_collection, clues_left=ALLOWED_ROUNDS - 1, player_cache={}, adversary_cache={}

    Returns the clue that does the best against the optimal player, and how many more guesses that would take.
    '''
    word_list = clue_collection.word_list
    if not word_list:
        # Somehow ruled out all options
        raise Exception(f'illegal word list {clue_collection}')
    if guess not in scrabble_five_letters:
        raise Exception(f'illegal guess {guess}')
    if clues_left == 1:
        # Adversary can pick any word user didn't guess, since they don't have any guesses left.
        if len(word_list) > 1:
            for word in word_list:
                if word != guess:
                    clue = give_clue(guess, word)
                    adversary_cache[(guess, clue_collection)] = (clue, math.inf, clues_left, word)
    # if len(word_list) == 1:
    #     for word in word_list:
    #         clue = give_clue(guess, word)
    #         cache[clue_collection] = (
    #             clue,
    #             optimal_player(clue, clue_collection, word_list, guesses_left=clues_left - 1))
    #         )

    if clue_collection in adversary_cache:
        clue, num, cache_clues_left, reference = adversary_cache[(guess, clue_collection)]
        if ((clues_left <= cache_clues_left and num == math.inf) # guarantee to stump player
            or clues_left >= num):                               # guarantee to be guessed
            return clue, num
        # Otherwise, recalculate clue under new clues_left number.

    best_clue = give_clue(guess, guess)
    best_clue_num = 0
    best_reference = guess

    # This one can't be five_letters, since server can't contradict itself.
    pbar = word_list if nested else tqdm(word_list, leave=False)
    for reference in pbar:
        if not nested:
            pbar.set_description(f"server considering {reference}")
        clue = give_clue(guess, reference)
        # Too slow if we consider all scrabble five letters, assume
        # player only guesses wordle words.
        next_guess, num = optimal_player(clue, clue_collection, guesses_left=clues_left - 1, player_cache=player_cache, adversary_cache=adversary_cache, nested=True, consider_guesses=clue_collection.word_list
                                         # scrabble_five_letters
                                         )
        if num > best_clue_num:
            best_clue = clue
            best_clue_num = num
            best_reference = reference
        if num > clues_left:
            # Can stump optimal player
            break

    # print(f'Adversarial clue for {guess} in:\n'
    #       f'{clue_collection} ({len(word_list)} possibilities):\n'
    #       f'{best_clue} ({best_clue_num} guesses left)')
    # return best_clue, best_clue_num
    LOG.debug(f'best reference was {best_reference}')
    adversary_cache[(guess, clue_collection)] = (best_clue, best_clue_num, clues_left, best_reference)
    return adversary_cache[(guess, clue_collection)][:2]

from tqdm import tqdm
def greedy_player(clue, clue_collection, word_list, yellow_multiplier=5, black_multiplier=20, precalculated='SALES'):
    '''
    Given guess and word_list, chooses the guess that results in the highest expected heuristic value.  The heuristic estimates that a green letter is worth 4 yellow letters and 20 black letters.

    Returns a tuple with two elements (1) the guess that has the highest expected heuristic value and (2) the dict mapping guesses to expected heuristic values.
    '''
    if not word_list:
        # Somehow ruled out all options
        raise Exception(f'illegal word list {clue_collection}\n{clue}')
    if not clue and word_list == five_letters and precalculated:
        # Use precalculated first guess to save effort.
        return precalculated, None
    if clue and clue.all_green():
        # Guessed the right word last time
        return clue.guess, None
    if len(word_list) == 1:
        # Only one option left
        for word in word_list:
            return word, None

    clue_collection = clue_collection.copy()
    if clue:
        clue_collection.add_clue(clue)
        word_list = clue.filter_words(word_list)

    best_guess = None
    best_guess_heuristic = 0

    def calc_heuristic(clue_collection):
        return (sum(bool(x) for x in clue_collection.greens) +
            sum(len(x) for x in clue_collection.yellows) / yellow_multiplier +
            len(clue_collection.blacks) / black_multiplier)
    # baseline = calc_heuristic(clue_collection)
    def calc_heuristic_after_clue(clue):
        next_col = clue_collection.copy()
        next_col.add_clue(clue)
        return calc_heuristic(next_col)

    info = {}
    baseline = calc_heuristic(clue_collection)
    # Could restrict search space to non-eliminated words (word_list)
    # for an approximation of hard mode, or all words (five_letters) for
    # normal mode.  hard mode should be faster, eliminating lots of options.
    guesses = word_list
    for guess in guesses:
        heuristics = []
        for actual in guesses:
            heuristic = calc_heuristic_after_clue(give_clue(guess, actual)) - baseline
            heuristics.append(heuristic)
        ave = np.mean(heuristics)
        info[guess] = ave
        if ave > best_guess_heuristic:
            best_guess = guess
            best_guess_heuristic = ave

    return (best_guess, info)


class GreedyPlayer():

    def __init__(self, coll, word_list, precalculated='SALES'):
        self.coll = coll
        self.word_list = word_list
        self.precalculated = precalculated

    def next_guess(self, clue):
        guess, _ = greedy_player(clue, self.coll, self.word_list, precalculated=self.precalculated)
        if clue:
            self.coll.add_clue(clue)
            self.word_list = clue.filter_words(self.word_list)
        return guess


# def test1():
#     test_list = ['AAAAA', 'BBBBB', 'AAABB', 'BBBAA']
#     test_cache = {}
#     test_cache2 = {}
#     assert (None, math.inf) == optimal_player(None, ClueCollection(), test_list, guesses_left=1, player_cache=test_cache, adversary_cache=test_cache2)

#     assert len(test_cache2) == 4

# test1()

# def test2():
#     # Asymmetrical, B's give more info.
#     test_list = ['AAAAA', 'BBBBB', 'CCCCC',
#                  'AAAAB', 'AAABB', 'AABBB', 'ABBBB',
#                  'BBBBC', 'BBBCC', 'BBCCC', 'BCCCC',]
#     test_cache = {}
#     test_cache2 = {}
#     strat, num = optimal_player(None, ClueCollection(), test_list, guesses_left=3, player_cache=test_cache, adversary_cache=test_cache2)
#     # Don't know why this fails now
#     return
#     assert strat == 'ABBBB'
#     assert num == 2
#     # for (g, l), (c, n) in test_cache2.items():
#     #     print(f'For:\n{l}\n\nGuess: {g}\nNext clue ({n} guesses left):\n', c)
#     #     print('---------------')

# test2()

# def test3():
#     # Prefer to guess non-repeated letters.
#     test_list = ['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE', 'ABCDE']
#     test_cache = {}
#     test_cache2 = {}
#     strat, num = optimal_player(None, ClueCollection(), test_list, guesses_left=7, player_cache=test_cache, adversary_cache=test_cache2)
#     assert strat == 'ABCDE'
#     assert num == 2
#     # for (g, l), (c, n) in test_cache2.items():
#     #     print(f'For:\n{l}\n\nGuess: {g}\nNext clue ({n} guesses left):\n', c)
#     #     print('---------------')

# test2()

def calculate_three_moves(three_move_cache={}, three_move_cache_adversary={}):
    strat, num = optimal_player(None, ClueCollection(), five_letters, guesses_left=3, player_cache=three_move_cache, adversary_cache=three_move_cache_adversary)
    strat, num
    len(three_move_cache_adversary)

    import dill

    with open('company_data.pkl', 'wb') as outp:
        dill.dump(optimal_player, outp)
        dill.dump(adversarial_server, outp)
    return adversary_cache


class Game():

    def __init__(self, word_list= five_letters, guesses=6):
        self.guesses_left = guesses
        self.clue_collection = ClueCollection()


class StaticGame(Game):

    def __init__(self, reference, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference
        self.success = False
        self.initial_guesses = self.guesses_left

    def guess(self, word):
        '''
        Return clue if any guesses left, False otherwise.
        '''
        if self.guesses_left <= 0:
            print('No guesses left.')
            print(f'The remaining words are {self.clue_collection.word_list}')
            return False
        if self.success:
            print('Already got the word.')
            return False
        try:
            clue = give_clue(word, self.reference)
            self.clue_collection.add_clue(clue)
            # print(self.clue_collection)
            self.guesses_left -= 1
            if word == self.reference:
                self.success = True
            return clue
        except Exception as e:
            raise e

    def check_success(self):
        '''
        Returns success and number of guesses it took if applicable.
        '''
        return (self.success,
                self.initial_guesses - self.guesses_left
                if self.success else None)


class AdversarialGame(Game):

    def __init__(self, player_cache={}, adversary_cache={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player_cache = player_cache
        self.adversary_cache = adversary_cache

    def guess(self, word):
        '''
        This will hang.
        '''
        if self.guesses_left <= 0:
            print('No guesses left.')
        try:
            clue, num = adversarial_server(word, self.clue_collection, clues_left=self.guesses_left, player_cache=self.player_cache, adversary_cache=self.adversary_cache)
            if num == math.inf:
                print("You're not going to get it.")
            if num == 0:
                print('You win! Yay :)')
            self.clue_collection.add_clue(clue)
            print(self.clue_collection)
            self.guesses_left -= 1
        except Exception as e:
            print(e)
        if self.guesses_left == 0:
            print(f'The remaining words are {self.clue_collection.word_list}')

def play(clues):
    '''
    Input: list of (text, greens, yellows) tuples
    Output: leftover words after applying clues constructed from those tuples
    '''
    today = five_letters
    for text, greens, yellows in clues:
        clue = Clue(text.upper(), greens, yellows)
        today = clue.filter_words(today)
    return today


class output_controller():
    def __init__(self):
        self.info_ = True
        self.stat_ = True
        self.prompt_ = True

    def info(self, *args, **kwargs):
        if self.info_:
            print(*args, **kwargs)

    def stat(self, *args, **kwargs):
        if self.stat_:
            print(*args, **kwargs)

    def prompt(self, *args, **kwargs):
        if self.prompt_:
            print(*args, ':', **kwargs)


out = output_controller()

def interactive_add_clue(game):
    guess = ''
    while len(guess) != 5:
        guess = input('Input clue text: ').upper()
    colors = ''
    while len(colors) != 5:
        colors = input('Input 5 colors (y=yellow and g=green): ').lower()

    greens = [i for i in range(len(colors)) if colors[i] == 'g']
    yellows = [i for i in range(len(colors)) if colors[i] == 'y']

    clue = Clue(guess, greens, yellows)
    game.clue_collection.add_clue(clue)
    out.info(f"Clue added: {clue}")
    indented_clue_collection = '\n'.join(['\t' + line for line in str(game.clue_collection).split('\n')])
    out.info(f"New game:\n{indented_clue_collection}")

def enter_interactive_game():
    game = AdversarialGame()
    player_cache = {}
    adversary_cache = {}
    while True:
        out.prompt('Input guess or command')
        sentence = input().upper()
        if not sentence.strip() or sentence.strip() in ['exit', 'quit', 'QUIT', 'q', '\\ex', '\\q']:
            out.prompt("Quit? Press 'enter' ⏎ again to confirm, or input sentence to continue")
            sentence = input()
            if not sentence.strip():
                out.info('Goodbye!')
                break
        if sentence.strip().lower() in ['help', 'h', '?']:
            out.info("Options: sentence, best guess, view clues, words left, add clue, restart, help, quit")
            continue
        if sentence.strip().lower() in ['best guess', 'b', 'g']:
            best_guess, num = optimal_player(None, game.clue_collection, game.guesses_left, game.player_cache, game.adversary_cache)
            out.info("Your best guess is:", best_guess, "with", num, "guesses remaining")
            response, num = adversarial_server(best_guess, game.clue_collection, game.guesses_left - 1, game.player_cache, game.adversary_cache)
            out.info("The worst case response would be", response)
            continue
        if sentence.strip().lower() in ['view clues', 'v']:
            out.info(game.clue_collection, '', sep='\n')
            continue
        if sentence.strip().lower() in ['words left', 'w', 'l']:
            out.info(game.clue_collection.word_list, '', sep='\n')
            continue
        if sentence.strip().lower() in ['add clue', 'a', 'c']:
            interactive_add_clue(game)
            continue
        if sentence.strip().lower() in ['restart', 'r']:
            game = AdversarialGame(player_cache=player_cache, adversary_cache=adversary_cache)
            continue

        game.guess(sentence)


def gametest():
    game = AdversarialGame()
    game.guess('HEART')
    game.guess('ACHES')

    game.guess('CACHE')

    adversarial_server('HEART', ClueCollection(), five_letters, clues_left=3, player_cache=three_move_cache, adversary_cache=three_move_cache_adversary)[0]

def test_static_game():
    game = StaticGame('RATED')
    game.guess('HEART')
    game.guess('EATER')
    game.guess('RATES')
    game.guess('RATED')
    assert (True, 4) == game.check_success()
    game.guess('RATED')
    assert (True, 4) == game.check_success()

# test_static_game()

def test_no_repeat_probabilistic():
    # Prefer to guess non-repeated letters.
    test_list = ['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE', 'ABCDE']
    test_cache = {}
    test_cache2 = {}
    strat, num = greedy_player(None, ClueCollection(), test_list)
    assert strat == 'ABCDE'
    # for (g, l), (c, n) in test_cache2.items():
    #     print(f'For:\n{l}\n\nGuess: {g}\nNext clue ({n} guesses left):\n', c)
    #     print('---------------')


def find_first_greedy_guess():
    a, b = greedy_player(None, ClueCollection(), five_letters)

    aves = {}
    for key in b:
        ave = np.mean(list(b[key].values()))
        aves[key] = ave
    with open('wordle_aves_5_yellow', 'w') as f:
        for word, ave in sorted(b.items(), key=lambda x: x[1]):
            f.write(f'{word}\t{ave:.2f}\n')

    print(f'Greedy guess: {a}')

def find_greedy_player_ave_turns(precalculated='SALES'):
    turns = {}
    for ref in tqdm(five_letters, leave=False):
        game = StaticGame(ref)
        player = GreedyPlayer(ClueCollection(), five_letters, precalculated)
        next_guess = player.next_guess(None)
        while not game.check_success()[0]:
            clue = game.guess(next_guess)
            if not clue:
                break
            next_guess = player.next_guess(clue)
        success, this_turns = game.check_success()
        if success:
            turns[ref] = this_turns
        else:
            turns[ref] = 100
    return turns

# turns = find_greedy_player_ave_turns()

# print(turns)

# clue = Clue('TARES', [], [0], [1,2,3,4])
# cluecoll = ClueCollection()
# cluecoll.add_clue(clue)

# guess, blah = greedy_player(Clue('POUTY', [], [3], [0, 1, 2, 4]), cluecoll, today)

# guess

# for word, num in reversed(sorted(blah.items(), key=lambda x: x[1])):
#     if len(set(word)) == 5 and word not in {'PITHY'}:
#         print(word, num)
#         break

clue = Clue('HEART', [], [3], [0,1,2,4])
today = clue.filter_words(five_letters)
len(today)
clue2 = Clue('POUTY', [], [3], [0,1,2,4])
today = clue2.filter_words(today)
len(today)
today

clue3 = Clue('CRUMB', [1,2], [], [0,3,4])
today = clue3.filter_words(today)
len(today)
today

clue4 = Clue('DRUNK', [1,2,3], [], [0,4])
today = clue4.filter_words(today)
len(today)
today

def test_clue_GG():
    clue = Clue('AAXXX', [0,1],[])
    today = {'AAAAA','ABAAA','BBAAA','AABBC'}
    assert clue.filter_words(today) == {'AAAAA','AABBC'}
def test_clue_GY():
    clue = Clue('AAXXX', [0],[1])
    today = {'AAAAA','ABAAA','BAAAA','BBAAA','AABBC'}
    assert clue.filter_words(today) == {'ABAAA'}
def test_clue_GB():
    clue = Clue('AAXXX', [0],[])
    today = {'AAAAA','ABAAA','BBAAA','AABBC','ABBBB','BABBB','BBAAA'}
    assert clue.filter_words(today) == {'ABBBB'}
def test_clue_GBY():
    clue = Clue('AAAXX', [0],[2])
    today = {'AAAAA','ABAAA','BBAAA','AABBC','ABABC','ABBAC','ABBBB','ABBCA','BBBAA'}
    assert clue.filter_words(today) == {'ABBCA','ABBAC'}

test_clue_GG()
test_clue_GY()
test_clue_GB()
test_clue_GBY()

if __name__ == '__main__':
    enter_interactive_game()
