import numpy as np

class TileCoding(object):
    def __init__(self, min_values, max_values, n_tilings, tile_frac):
        self.min_values = min_values
        self.max_values = max_values
        self.n_tilings = n_tilings
        self.tile_frac = tile_frac
        self.total_n_tiles = 0
        self.n_tiles_tile_set = []
        self.init_tilings()
    
    
    def init_tilings(self):
        self.all_tilings = []
        for dim in range(len(self.min_values)):
            self.all_tilings.append(self.create_tiling(dim))
        

    def create_tiling(self, dim):
        tile_width = self.tile_frac * (self.max_values[dim] - \
                     self.min_values[dim])
        offset = tile_width / self.n_tilings
        start_pos = self.min_values[dim] - (self.n_tilings-1)*offset
        n_tiles = int(np.ceil(abs(start_pos - self.max_values[dim])/tile_width)) + 1
        self.total_n_tiles += (n_tiles) * self.n_tilings
        self.n_tiles_tile_set.append(n_tiles)
        tilings = []
        
        for i in range(self.n_tilings):
            tilings.append([start_pos + i*offset + tile_width*x \
                            for x in range(n_tiles)])
        return tilings


    def get_tile_code(self, state):
        working_idx = 0
        vec = np.zeros(self.total_n_tiles)
        
        for i in range(len(state)):
            for tiling in self.all_tilings[i]:
                tile_idx = np.digitize(state[i], tiling)
                vec[working_idx + tile_idx] = 1
                working_idx += self.n_tiles_tile_set[i]
        return vec
            
        
    def get_feature_vector(self, state, action):
        tile_code = self.get_tile_code(state)
        return np.append(tile_code, action)
    
    
    def get_feature_vectors_for_actions(self, state, n_actions):
        feature_vectors = [self.get_feature_vector(state, action) for action\
                           in range(n_actions)]
        return feature_vectors
        
        
        
    
    
    
    
import gym


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    min_values = [env.min_position, -env.max_speed]
    max_values = [env.max_position, env.max_speed]
    n_tilings = 8
    tile_frac = 1/8
    obj = TileCoding(min_values, max_values, n_tilings, tile_frac)

    











