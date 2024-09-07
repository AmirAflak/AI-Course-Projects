from Agents import Agent
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(state, 0, 0)[1]

    def maxValue(self, gameState, depth, agentIndex):
        if gameState.isGameFinished() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        v = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.minValue(successor, depth, 1)[0]
            if value > v:
                v, bestAction = value, action
        return v, bestAction

    def minValue(self, gameState, depth, agentIndex):
        if gameState.isGameFinished():
            return self.evaluationFunction(gameState), None
        v = float("inf")
        bestAction = None
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            depth += 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  
                value = self.maxValue(successor, depth, nextAgent)[0]
            else:
                value = self.minValue(successor, depth, nextAgent)[0]
            if value < v:
                v, bestAction = value, action
        return v, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")

        return self.maxValue(gameState, 0, 0, alpha, beta)[1]
    
    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isGameFinished() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        v = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.minValue(successor, depth, 1, alpha, beta)[0]
            if value > v:
                v, bestAction = value, action
            if v > beta:    
                return v, bestAction
            alpha = max(alpha, v)
        return v, bestAction

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isGameFinished():
            return self.evaluationFunction(gameState), None
        v = float("inf")
        bestAction = None
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            depth += 1
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                value = self.maxValue(successor, depth, nextAgent, alpha, beta)[0]
            else:
                value = self.minValue(successor, depth, nextAgent, alpha, beta)[0]
            if value < v:
                v, bestAction = value, action
            if v < alpha:
                return v, bestAction
            beta = min(beta, v)
        return v, bestAction
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0, 0)[1]
    
    def maxValue(self, gameState, depth, agentIndex):
        if gameState.isGameFinished() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        v = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.expectValue(successor, depth, 1)[0]
            if value > v:
                v, bestAction = value, action
        return v, bestAction

    def expectValue(self, gameState, depth, agentIndex):
        if gameState.isGameFinished():
            return self.evaluationFunction(gameState), None
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState), None
        prob = 1.0 / len(actions)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            if nextAgent == 0:
                v += prob * self.maxValue(successor, depth, nextAgent)[0]
            else:
                v += prob * self.expectValue(successor, depth, nextAgent)[0]
        return v, None



def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    "*** YOUR CODE HERE ***"
    playerIndex = currentGameState.index


    # parity
    myScore = currentGameState.getScore(playerIndex)
    opponentScores = [currentGameState.getScore(i) for i in range(currentGameState.getNumAgents()) if i != playerIndex]
    parity = myScore - max(opponentScores)

    # corners
    corners = currentGameState.getCorners()
    myCorners = sum([1 for corner in corners if corner == playerIndex])
    cornerControl = myCorners / 4  

    # mobility  
    mobility = len(currentGameState.getLegalActions(playerIndex))

    # stability
    stability = calculateStability(currentGameState, playerIndex)
    
    score = (parity * 1.0) + (cornerControl * 2.0) + (mobility * 0.5) + (stability * 0.5)
    return score

def calculateStability(gameState, playerIndex):
    stable_pieces = 0
    for piece in gameState.getPieces(playerIndex):
        if isStable(piece, gameState, playerIndex):
            stable_pieces += 1
    return stable_pieces

def isStable(piece, gameState, playerIndex):
    if piece in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        return True

    if piece[0] in [0, 7] or piece[1] in [0, 7]:
        return True

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dx, dy in directions:
        adjacent = (piece[0] + dx, piece[1] + dy)
        if not isOnBoard(adjacent) or gameState.getPiece(adjacent) != playerIndex:
            return False
    return True

def isOnBoard(position):
    x, y = position
    return 0 <= x < 8 and 0 <= y < 8  


# Abbreviation
better = betterEvaluationFunction