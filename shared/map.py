import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

class Map(object):
    def __init__(self, observation, columns):
        self.columns = columns

        self.observation = observation
        self.player_index = observation.index
        self.player_head = observation.geese[self.player_index][0]

        self.geese = []
        for i in range(len(observation.geese)):
            buffer = []
            for j in range(len(observation.geese[i])):
                buffer.append(self.coord_to_abs(*self.translate(observation.geese[i][j])))
            
            self.geese.append(buffer)
            
    def translate(self, position):
        row, column = row_col(position, self.columns)

        head_row, head_column = row_col(self.player_head, self.columns)

        return (row - head_row + 3) % 7, (column - head_column + 5) % 11
    
    def coord_to_abs(self, row, column):
        return (row * self.columns + column) % 77
    
    def get_heads_tails(self):
        output = []

        backup = [38] #38 is the middle
        for i in range(4):
            if i != self.player_index:
                if len(self.observation.geese[i]) != 0:
                    backup = [self.observation.geese[i][0]]

        for i in range(4):
            if i != self.player_index:
                goose = self.observation.geese[i]
                if len(goose) == 0:
                    goose = backup

                output.append(self.translate(goose[0]))
                output.append(self.translate(goose[-1]))
        
        # Add our own tail
        output.append(self.translate(self.observation.geese[self.player_index][-1]))

        return output
    
    def occupied(self, position):
        for i in range(4):
            if position in self.geese[i][:-1]:
                return True
        
        return False

    def build_opponent_map(self, index):
        output = np.zeros(77) # 77 is the boardsize, thats 7x11
        opponent = self.geese[index]

        if (len(opponent) == 0):
            return output

        for i in range(len(opponent) - 1):
            output[i] = 1

        new_pos = [opponent[0] - 1, opponent[0] + 1, opponent[0] - self.columns, opponent[0] + self.columns]
        food_count = 0
        possible_positions = []

        for pos in new_pos:
            if pos in self.observation.food:
                food_count += 1

            if not self.occupied(pos):
                possible_positions.append(pos)
        
        if (len(possible_positions) != 0):
            possible_positions = np.array(possible_positions) % 77
            output[opponent[-1]] = food_count / len(possible_positions)
            output[possible_positions] += 1 / len(possible_positions)

        return output
    
    def build_maps(self):
        output = []

        for i in range(4):
            if i != self.player_index:
                output.append(self.build_opponent_map(i))
        
        player_map = np.zeros(77)
        player = self.geese[self.player_index]
        for i in range(len(player)):
            player_map[player[i]] = 1
        
        output.append(player_map)
        return np.concatenate(output)
            

