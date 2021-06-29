import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

class Map(object):
    def __init__(self, observation, columns):
        self.columns = columns

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

    def translate(self, position, zero_center=False):
        row, column = row_col(position, self.columns)

        head_row, head_column = row_col(self.player_head, self.columns)

        if zero_center:
            return (row - head_row + 3) % 7 - 3, (column - head_column + 5) % 11 - 5

        return (row - head_row + 3) % 7, (column - head_column + 5) % 11
    
    def coord_to_abs(self, row, column):
        return (row * self.columns + column) % 77
        
    def occupied(self, position):
        for i in range(4):
            if position in self.geese[i][:-1]:
                return True
        
        return False

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

        head[opponent[0]] = 1

        for i in range(len(opponent)):
            body[opponent[i]] = 1

        tail[opponent[-1]] = 1

        return head, body, tail
    
    def build_maps(self):
        output = []

        opponent_order = np.arange(4)
        np.random.shuffle(opponent_order)
        for i in opponent_order:
            if i != self.player_index:
                output.extend(self.build_opponent_map(i))
        
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

        return output
            

