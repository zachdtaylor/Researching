# The AI class uses the neural network. The Computer class uses Q tables.

# coding: utf-8

import numpy as np
import pandas as pd
import random
import copy
from __future__ import division
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, core

board = "| {0} | {1} | {2} |\n-------------\n| {3} | {4} | {5} |\n-------------\n| {6} | {7} | {8} |"

model = Sequential()

model.add(Dense(output_dim=30, input_dim=9))
model.add(Activation("relu"))
model.add(Dense(output_dim=15))
model.add(Activation('relu'))
model.add(Dense(output_dim=9))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

def tie(state):
    if 0 not in state and not win(state,1) and not win(state,2): 
        return True
    else: return False
    
def win(state, token):
    if token == state[0] == state[1] == state[2]: return True
    if token == state[3] == state[4] == state[5]: return True
    if token == state[6] == state[7] == state[8]: return True
    if token == state[0] == state[3] == state[6]: return True
    if token == state[1] == state[4] == state[7]: return True
    if token == state[2] == state[5] == state[8]: return True
    if token == state[0] == state[4] == state[8]: return True
    if token == state[6] == state[4] == state[2]: return True

class Computer(object):
    def __init__(self):
        self.Q = {} # Dictionary will be formatted as {((state), action): q-value}. 

    def legalRand(self, state): # Returns a random legal move in the state given
        possible = []
        for i in range(9):
            if state[i] == 0:
                possible.append(i)
        move_index = random.choice(possible)
        return move_index
    
    def legal(self, state): # Returns a list of all legal moves in the state given
        legal = []
        for i in range(9):
            if state[i] == 0:
                legal.append(i)
        return legal
    
    def epsilon_greedy(self, epsilon, state): # Makes a random move with probability epsilon
        stateA = copy.copy(state)             # Makes the best learned move with probability 1-epsilon
        if random.random() > epsilon:
            move = self.best_move(state)
        else:
            move = self.legalRand(state)
        stateA[move] = 1
        return move, stateA # Returns the move and the new state after having made the move
    
    def learn(self, games = 10, lrate = 1, discfac = .5, epsilon = .1):
        for i in range(games):
            state = [0,0,0,0,0,0,0,0,0] 
            while True:
                move, stateA = self.epsilon_greedy(epsilon, state)
                if win(stateA, 1): # The agent is rewarded if it wins
                    state = tuple(state)
                    if (state,move) in self.Q: self.Q[(state,move)] += lrate*(100+discfac*(100)-self.Q[(state,move)]) 
                    if (state,move) not in self.Q: self.Q[(state,move)] = 0 
                    break
                if 0 not in stateA: # The agent is punished if the game ends up in a tie
                    state = tuple(state)
                    if (state,move) in self.Q: self.Q[(state,move)] += lrate*(0+discfac*(-100)-self.Q[(state,move)])
                    if (state,move) not in self.Q: self.Q[(state,move)] = 0 
                    break
                else:
                    randmove = self.legalRand(stateA)
                    stateA[randmove] = 2
                    if win(stateA, 2): # The agent is also punished if the opponent wins
                        state = tuple(state)
                        if (state,move) in self.Q: self.Q[(state,move)] += lrate*(0+discfac*(-100)-self.Q[(state,move)]) 
                        if (state,move) not in self.Q: self.Q[(state,move)] = 0 
                        break
                    else: # Otherwise, updates to Q values are made normally
                        state = tuple(state)
                        if (state,move) in self.Q: self.Q[(state,move)] += lrate*(0+discfac*(max(self.nextQs(stateA)))-self.Q[(state,move)])
                        if (state,move) not in self.Q: self.Q[(state,move)] = 0
                state = stateA
        print("Done.")
        
    def nextQs(self, state): # Returns a list of Q values associated with each possible move in the state given
        possible_moves, q = self.legal(state), []
        state = tuple(state)
        for i in possible_moves:
            if (state,i) in self.Q: q.append(self.Q[(state,i)])
            else: q.append(0)
        return q
    
    def best_move(self, state):
        possible_moves, q = self.legal(state), self.nextQs(state)
        count = q.count(max(q))
        if count > 1: # If there is more than one best move, randomly choose one
            best_choices = [k for k in range(len(possible_moves)) if q[k] == max(q)]
            move_index = random.choice(best_choices)
        else: move_index = q.index(max(q)) # Otherwise, choose the best option
        return possible_moves[move_index]
        
    def play(self, state):
        move = self.best_move(state)
        print("Computer's move: {0}".format(move+1))
        return move
    
    def getType(self):
        return "Computer"

C = Computer()
C.learn(games = 225000, lrate = .2, discfac = 1, epsilon = .1) 

class AI:
    def __init__(self):
        self.model = model

    def legalRand(self, state): # Returns a random legal move in the state given
        possible = []
        for i in range(9):
            if state[i] == 0:
                possible.append(i)
        move_index = random.choice(possible)
        return move_index
    
    def legal(self, state): # Returns a list of all legal moves in the state given
        legal = []
        for i in range(9):
            if state[i] == 0:
                legal.append(i)
        return legal
    
    # Returns max legal q value
    def legalQ(self, Qvals, state):
        q = []
        legal_moves = self.legal(state)
        for i in legal_moves: q.append(Qvals[0][i])
        return max(q)
    
    # Makes a random move with probability epsilon
    # Makes the best learned move with probability 1-epsilon
    def epsilon_greedy(self, epsilon, state):
        stateA = copy.copy(state)         
        if random.random() > epsilon:
            move = self.best_move(state)
        else:
            move = self.legalRand(state)
        stateA[move] = 1
        return move, stateA # Returns the move and the new state after having made the move
    
    def learn(self, games = 10, discfac = .5, epsilon = .1):
        games_won = 0
        counter = 0
        
        for i in range(games):
            # Each game starts with an empty board
            state = np.array([0,0,0,0,0,0,0,0,0])
            
            while True:
                # Get the NN's output for the current state
                qvals = model.predict(state.reshape(1,9), verbose=0)
                
                # Make the best move (highest Q value) with probability 1-epsilon
                # and get the new state after move is made
                move, new_state = self.epsilon_greedy(epsilon, state)
                
                # If the AI wins after this move, or causes a tie, break out of the loop
                # to train NN
                if win(new_state, 1) or tie(new_state): break
                
                # If we're at this point, the AI didn't win and didn't tie, so
                # the opponent makes its move
                #if (random.random() >= .5): opp_move = self.legalRand(new_state)
                opp_move = C.best_move(new_state)
                
                # Put a 2 in the state where the opponent made their move
                new_state[opp_move] = 2
                # Then if the opponent wins, break to train NN
                if win(new_state, 2): break
                
                # By this point, we know neither the AI or opponent has won,
                # so we train NN 
                else:
                    # Reward: 0 since game isn't over
                    reward = self.getReward(new_state)
                    # Get q values on the state now with the new moves made
                    newQ = model.predict(new_state.reshape(1,9), verbose=0)
                    # Get max legal q value
                    maxQ = self.legalQ(newQ, new_state)
                    
                    # vector y = qvals, differing only in the spot where the AI
                    # chose to move. There, y holds discfac * maxQ. The purpose of this
                    # is to train each move based on the best option the AI has in 
                    # the next state. In other words, moves get judged based on future
                    # consequences 
                    update = reward + (discfac * maxQ)
                    y = np.zeros((1,9))
                    y[:] = qvals[:] 
                    y[0][move] = update
                    #print("Move: %r" % move)
                    #print("Update: %r" % update)
                    #print(y)
                    # 10 epochs to make sure the behavior is enforced 
                    self.model.fit(qvals, y, nb_epoch=1, verbose=0)
                    # set the current state to the new state
                    state = new_state
            
            # Rewards: win = 10, loss = -10, tie = -10
            reward = self.getReward(new_state)
            # When the game is over, we train soley based on whether the AI won, lost, or tie
            y = np.zeros((1,9))
            y[:] = qvals[:]
            y[0][move] = reward
            #print("Move: %r" % move)
            #print("Reward %s" % reward)
            #print(y)
            model.fit(qvals, y, nb_epoch=1, verbose=0)
            
            # As training progresses, we make it more likely that AI chooses the best moves
            if epsilon > .1: epsilon -= 1/games
            if (reward == 10): games_won += 1
            counter += 1
            if counter % 10000 == 0: 
                print ("Game {0} of {1} complete.".format(counter, games))
                print("Epsilon: ", epsilon)
        print ("Done. Won {0} of {1} games.".format(games_won, games))
    
    def getReward(self, state):
        if win(state, 1):
            return 10
        elif win(state, 2) or tie(state):
            return -10
        else:
            return 0
    
    # Returns a list of Q values associated with each possble move in the state given
    def nextQs(self, state): 
        state = np.asarray(state)
        Q, q = model.predict_proba(state.reshape(1,9), verbose=0), []
        legal_moves = self.legal(state)
        for i in legal_moves: q.append(Q[0][i])
        return q
    
    def best_move(self, state):
        legal_moves, q = self.legal(state), self.nextQs(state)
        #print("Legal moves: {0} q: {1}".format(legal_moves, q))
        count = q.count(max(q))
        if count > 1:
            best_choices = [k for k in range(len(legal_moves)) if q[k] == max(q)]
            move_index = random.choice(best_choices)
        else: move_index = q.index(max(q))
        #print "Move_index: %d" % move_index
        return legal_moves[move_index]
        
    def play(self, state):
        move = self.best_move(state)
        print ("Computer's move: {0}".format(move+1))
        return move
    
    def getType(self):
        return "Computer"

class Human:
    def getType(self):
        return "Human"

    def play(self, state):
        move = int(input("Your move: "))
        if move >0 and move < 10:
            return move-1
        else:
            raise ValueError("Entry must be a number between 1 and 9")

class TicTacToe:
    def play(self, player1, player2):
        state = [0 for j in range(9)]
        start = [' ' for i in range(9)]
        print (board.format(*start), '\n')
        while True:
            player1_move = player1.play(state)
            start[player1_move], state[player1_move] = 'X', 1
            print (board.format(*start), '\n')
            if win(state, 1): 
                player1_type = player1.getType()
                print("%s wins!" % player1_type)
                break
            if tie(state):
                print ("Tie.")
                break
            player2_move = player2.play(state)
            start[player2_move], state[player2_move] = 'O', 2
            print (board.format(*start), '\n')
            if win(state, 2):
                player2_type = player2.getType()
                print ("%s wins!" % player2_type)
                break

CPU = AI()
CPU.learn(games = 10000, discfac = .9, epsilon = 0)

Me = Human()
T = TicTacToe()
T.play(CPU, Me)

model.predict(np.array([0,0,0,2,0,1,0,2,1]).reshape(1,9))
