import numpy as np
import cv2

rounds = 1000
imgSize = 101
boardSize = 3
player1 = 1
player2 = 2
blank = 0
lr = 0.01
discount = 0.99
debug = True


class Game:
    board = np.zeros([boardSize, boardSize])+blank
    scorePlayer1 = 0
    scorePlayer2 = 0
    scoreTies = 0
    imgPlayer1 = 0
    imgPlayer2 = 0
    imgBlank = 0

    def __init__(self):
        self.createImages()

    def clear(self):
        self.board = np.zeros([boardSize, boardSize])+blank

    def createImages(self):
        self.imgBlank = np.zeros([imgSize, imgSize]) + 1
        self.imgPlayer1 = np.zeros([imgSize, imgSize]) + 1
        self.imgPlayer2 = np.zeros([imgSize, imgSize]) + 1

        m = (imgSize-1)/2
        for i in range(self.imgPlayer2.shape[0]):
            for j in range(self.imgPlayer2.shape[1]):
                r = np.sqrt(np.power(i-m,2)+np.power(j-m,2))
                if np.round(r) == m:
                    self.imgPlayer2[i,j] = 0

        for i in range(self.imgPlayer1.shape[0]):
            self.imgPlayer1[i, i] = 0
            self.imgPlayer1[i, imgSize-i-1] = 0


    def move(self, block, player):
        row = np.round(np.mod(block,boardSize))
        col = np.floor(block/boardSize)
        self.board[row.astype(int), col.astype(int)] = player

    def boardImage(self):
        n = imgSize*3+2
        img = np.zeros([n, n]) + 1
        img[imgSize, :] = 0
        img[2*imgSize+1, :] = 0
        img[:, imgSize] = 0
        img[:, 2 * imgSize + 1] = 0
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                p = self.board[i, j]
                if p == player1:
                    curImage = self.imgPlayer1
                elif p == player2:
                    curImage = self.imgPlayer2
                else:
                    curImage = self.imgBlank
                img[i*imgSize+i:(i+1)*imgSize+i,j*imgSize+j:(j+1)*imgSize+j] = curImage
        return img

    def getScore(self):
        return "X: " + str(self.scorePlayer1) + " - O: " + str(self.scorePlayer2)

    def getBoardImage(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = self.boardImage()
        k = 50
        img = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_CONSTANT)
        img = cv2.putText(img, game.getScore(), (imgSize-5, k - 5), font, 1, (1, 1, 1))
        img = cv2.putText(img, "T: " + str(self.scoreTies), (imgSize+k, 4*imgSize-10), font, 1, (1, 1, 1))
        return img

    def isOver(self):
        if self.isWonPlayer(player1):
            self.scorePlayer1 +=1
            return True
        elif self.isWonPlayer(player2):
            self.scorePlayer2 +=1
            return True
        else:
            if self.possibleMoves().size == 0:
                self.scoreTies +=1
                return True
            return False

    def isWonPlayer(self, p):
        N = boardSize
        b = self.board == p
        sumColumns = np.sum(b == 1,0) == N
        sumRows = np.sum(b == 1, 1) == N
        if sumColumns.any():
            return True
        if sumRows.any():
            return True
        else:
            for i in range(N):
                if not b[i, i]:
                    break
                if i == N-1:
                    return True

            for i in range(N):
                if not b[i, N-i-1]:
                    break
                if i == N - 1:
                    return True

            return False

    def possibleMoves(self):
        list = []
        index = 0
        for j in range(self.board.shape[1]):
            for i in range(self.board.shape[0]):
                if self.board[i, j] == blank:
                    list.append(index)
                index += 1
        return np.asarray(list)

class Player:
    type = 0
    decay = 0
    explorationRate = 1
    q = {}
    lastState = []

    def __init__(self, type, decay):
        self.type = type
        self.decay = decay

    def play(self, game):
        moves = game.possibleMoves()
        explore = self.explorationRate >= np.random.rand()
        state = self.boardState(game.board)
        newState = state.copy()

        bestMove = -1
        bestValue = float('-Inf')
        for i in range(moves.size):
            curValue = self.getValueForMove(state, moves[i])
            if curValue > bestValue:
                bestValue = curValue
                bestMove = moves[i]

        if explore:
            block = moves[np.random.randint(0, moves.size)]
            game.move(block, self.type)
            newState[block] = self.type

        else:
            game.move(bestMove, self.type)
            newState[bestMove] = self.type

        self.lastState = newState.copy()

        curValue = self.getValue(state)

        self.explorationRate -= self.decay
        self.explorationRate = np.max([self.explorationRate, 0])

        if game.isWonPlayer(self.type):
            self.q[newState.__str__()] = 1
            self.lastState = []
            self.q[state.__str__()] += lr*(discount*1 - curValue)
            return 1
        else:
            self.q[state.__str__()] += lr*(discount*bestValue - curValue)
            return 0


    def lose(self):
        self.q[self.lastState.__str__()] = -1
        self.lastState = []

    def boardState(self, board):
        s = []
        for j in range(board.shape[1]):
            for i in range(board.shape[0]):
                s.append(board[i,j])
        return s

    def getValueForMove(self, state, move):
        curState = state.copy()
        curState[move] = self.type
        return self.getValue(curState)


    def getValue(self, state):
        s = state.__str__()
        if not self.q.keys().__contains__(s):
            self.q[s] = 0
        return self.q[s]

if __name__ == "__main__":
    game = Game()
    p1 = Player(player1, 0.0)
    p2 = Player(player2, 0.01)
    initVideo = True

    for i in range(rounds):
        turn = 1
        while True:
            if turn:
                if p1.play(game):
                    p2.lose()
            else:
                if p2.play(game):
                    p1.lose()
            turn = 1-turn
            img = game.getBoardImage()
            cv2.imshow('Game', img)
            cv2.waitKey(1)
            if initVideo:
                video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 48, img.shape)
                initVideo = False
            img8 = 255*img
            video.write(cv2.cvtColor(img8.astype(np.uint8), cv2.COLOR_GRAY2BGR))

            if game.isOver():
                break
        game.clear()

    cv2.destroyAllWindows()
    video.release()

