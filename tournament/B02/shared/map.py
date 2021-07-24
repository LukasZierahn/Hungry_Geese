import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

class Map(object):
    def __init__(self, observation, columns=11, rows=7, flipped_left_right=False, flipped_up_down=False, opponent_order=np.arange(4)):
        self.columns = columns
        self.rows = rows

        self.flipped_left_right = flipped_left_right
        self.flipped_up_down = flipped_up_down

        self.opponent_order = opponent_order

        self.observation = observation
        self.player_index = observation.index
        self.player_head = observation.geese[self.player_index][0]

        self.geese = []
        for i in np.arange(len(observation.geese)):
            buffer = []
            for j in range(len(observation.geese[i])):
                buffer.append(self.coord_to_abs(*self.translate(observation.geese[i][j])))
            
            self.geese.append(buffer)
        

        self.food = []
        for food in observation.food:
            self.food.append(self.coord_to_abs(*self.translate(food)))

    def transform_move_wrt_flipping(self, move):
        move = 1 if move == 3 and self.flipped_left_right else move
        move = 3 if move == 1 and self.flipped_left_right else move

        move = 0 if move == 2 and self.flipped_up_down else move
        move = 2 if move == 0 and self.flipped_up_down else move

        return move

    def translate(self, position):
        row, column = row_col(position, self.columns)

        head_row, head_column = row_col(self.player_head, self.columns)


        row = (row - head_row + 3) % self.rows
        column = (column - head_column + 5) % self.columns

        row += 2 * (3 - row) if self.flipped_up_down else 0
        column += 2 * (5 - column) if self.flipped_left_right else 0

        return row, column
    
    def coord_to_abs(self, row, column):
        return (row * self.columns + column) % 77
        
    def occupied(self, position):
        for i in range(4):
            if position in self.geese[i][:-1]:
                return True
        
        return False

    def get_suicide(self, observation, invalid_action):
        suicide = np.zeros(4, dtype=np.bool)

        map = Map(observation, 11)
        positions = map.generate_possible_pos(38) #38 is the middle, i. e. the player head
        for i in range(len(positions)):
            suicide[i] = map.occupied(positions[i])

        suicide[invalid_action] = True

        return suicide


    def generate_possible_pos(self, origin):
        (row, column) = row_col(origin, self.columns)

        return [self.coord_to_abs((row - 1) % 7, column), self.coord_to_abs(row, (column + 1) % 11), self.coord_to_abs((row + 1) % 7, column), self.coord_to_abs(row, (column - 1) % 11)]

    def build_opponent_map(self, index):
        head = np.zeros(77) # 77 is the boardsize, thats 7x11
        body = np.zeros(77)
        tail = np.zeros(77)
        opponent = self.geese[index]

        if (len(opponent) == 0):
            return head, body, tail

        for pos in self.generate_possible_pos(opponent[0]):
            head[pos] = 1

        for i in range(len(opponent)):
            body[opponent[i]] = 1

        tail[opponent[-1]] = 1

        return head, body, tail
    
    def build_maps(self):
        output = []
        opponent_maps = []

        for i in self.opponent_order:
            if i != self.player_index:
                opponent_maps.append(self.build_opponent_map(i))
        
        player_map = np.zeros(77)
        player = self.geese[self.player_index]
        for i in range(len(player)):
            player_map[player[i]] = 1
        
        output.append(player_map)

        player_tail_map = np.zeros(77)
        player_tail_map[player[-1]] = 1
        output.append(player_tail_map)

        # Food
        food = np.zeros(77)
        for food_pos in self.food:
            food[food_pos] = 1
        output.append(food)

        """for map in output:
            import matplotlib.pyplot as plt
            data = map.reshape(7,11)
            fig, ax = plt.subplots()
            # Using matshow here just because it sets the ticks up nicely. imshow is faster.
            ax.matshow(data, cmap='seismic')

            for (i, j), z in np.ndenumerate(data):
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

            plt.show()
        """

        return output, opponent_maps
            

