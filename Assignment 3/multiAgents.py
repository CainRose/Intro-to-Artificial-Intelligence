# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import print_function, division

from util import manhattanDistance
from game import Directions
import random, util
from collections import deque

from game import Agent
import searchAgents
import search

INF = 999999


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in
                  legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (food) and Pacman position after moving (pacman_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacman_pos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        old_food = currentGameState.getFood()
        capsules = successorGameState.getCapsules()
        old_capsules = currentGameState.getCapsules()
        ghost_states = successorGameState.getGhostStates()
        old_ghost_states = currentGameState.getGhostStates()
        scared_times = [g.scaredTimer for g in ghost_states]
        old_scared_times = [g.scaredTimer for g in old_ghost_states]

        food_score = sum([f for row in food for f in row])
        old_food_score = sum([f for row in old_food for f in row])
        capsule_score = len(capsules)
        old_capsule_score = len(old_capsules)
        food_distance = [manhattanDistance(pacman_pos, (i, j))
                         for i, row in enumerate(food)
                         for j, f in enumerate(row) if f]
        food_distance = min(food_distance) if food_distance else 0
        food_distance = -0.01 if old_food_score > food_score else food_distance
        capsule_distance = [manhattanDistance(pacman_pos, c)
                            for c in capsules]
        capsule_distance = min(capsule_distance) if capsule_distance else 0
        capsule_distance = -0.01 if old_capsule_score > capsule_score else capsule_distance
        action_score = (action == 'Stop') * 0.01

        enemy_dist = [manhattanDistance(pacman_pos, g.configuration.pos)
                      for g in ghost_states]
        for i, d in enumerate(enemy_dist):
            d = d if d != 0 else 0.00001
            d = 1 / d if d < 5 else 0
            d = 2 * d if scared_times[i] == 0 else -d
            if old_scared_times[i] - scared_times[i] > 1:
                d = -10
            enemy_dist[i] = d
        enemy_dist = sum(enemy_dist)

        score = food_distance + enemy_dist * 5 + \
                capsule_distance + action_score
        # print(action, food_distance, capsule_distance, enemy_dist,
        #       action_score, '--', score)
        return -score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        bestVal = -INF
        bestAction = None
        searchDepth = self.depth * gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            newVal = self.minimax(state, 1, searchDepth - 1)
            if newVal > bestVal:
                bestVal = newVal
                bestAction = action
        return bestAction

    def minimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:  # PACMAN
            best = -INF
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                best = max(best, self.minimax(state, next_agent, depth - 1))
        else:  # Ghosts
            best = INF
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                best = min(best, self.minimax(state, next_agent, depth - 1))
        return best


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestVal = -INF
        bestAction = None
        searchDepth = self.depth * gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            newVal = self.alphaBeta(state, 1, searchDepth - 1, bestVal, INF)
            if newVal > bestVal:
                bestVal = newVal
                bestAction = action
        return bestAction

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:  # PACMAN
            best = -INF
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                val = self.alphaBeta(state, next_agent, depth - 1, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        else:  # Ghosts
            best = INF
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                val = self.alphaBeta(state, next_agent, depth - 1, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        bestVal = -INF
        bestAction = None
        searchDepth = self.depth * gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            newVal = self.expectimax(state, 1, searchDepth - 1)
            if newVal > bestVal:
                bestVal = newVal
                bestAction = action
        return bestAction

    def expectimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:  # PACMAN
            best = -INF
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                best = max(best, self.expectimax(state, next_agent, depth - 1))
        else:  # Ghosts
            actions = gameState.getLegalActions(agentIndex)
            best = 0
            for action in actions:
                state = gameState.generateSuccessor(agentIndex, action)
                best += self.expectimax(state, next_agent, depth - 1)
            best /= len(actions)
        return best


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: There are three main evaluated aspects of the game state:

      Current Score: The only way that changes in state as a result of actions
      can be accounted for, e.g. eating a ghost, eating food

      Food Distance: The inverse distance to the nearest food, found using bfs.
      If there is food nearby, we should go for it.

      Enemy Distance: The inverse distance to each enemy. Only non-zero if the
      ghost is scared, to prioritize going for it.
    """

    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [g.scaredTimer for g in ghost_states]

    anyfood = searchAgents.AnyFoodSearchProblem(currentGameState)
    food_distance = search.bfs(anyfood)
    food_distance = 1 / len(food_distance) if food_distance else 0

    enemy_dist = [manhattanDistance(pacman_pos, g.configuration.pos)
                  for g in ghost_states]
    for i, d in enumerate(enemy_dist):
        d = d if d != 0 else 0.00001
        d = 1 / d if d < 5 or scared_times[i] != 0 else 0
        d = 0 if scared_times[i] == 0 else -d
        enemy_dist[i] = d
    enemy_dist = sum(enemy_dist)

    score = currentGameState.getScore()
    return score + 0.1*food_distance - enemy_dist


# Abbreviation
better = betterEvaluationFunction
