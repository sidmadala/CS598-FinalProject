from scipy.interpolate import interp1d
from scipy import signal, io
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os

# Number of particles (tunable)
numParticles = 500
# degree of the fitted polynomial
deg = 100
# Frequency of model per second
freq = 50

# Path for graphs
graphsFolder ="graphs/"

################################ Do Not Edit Below ###########################
channels = ["Left", "Right"] # audio channels


def Predict(time, particles, control, dt, sigma):
    # Predict forward in time all particles
    for index, particle in enumerate(particles):
        particle = particle + control(time)*dt + np.random.normal(0,sigma)
        # We are predicting forward essentially using Euler integration with
        # noise
        particles[index] = particle


    return particles

def Update(particles, mu, sigma):
    # New particles sampled. Importance sampling.
    # The probabilities of seeing each sample
    weights = np.zeros(particles.size)
    for i in range(particles.size):
        # Our observation model is normally distributed
        weights[i] = norm.pdf(particles[i], mu, sigma)

    if (np.isnan(np.linalg.norm(weights,1)) or np.linalg.norm(weights,1) == 0):
        # If all the probabilities are infinity or 0
        # Use uniform weighting
        weights = np.ones(particles.size)/particles.size
    else:
        # Otherwise, normalize the weights to form a valid probability
        # distribution
        weights = weights / np.linalg.norm(weights, 1)

    # Sample with replacement
    particles = np.random.choice(particles, size=particles.size, p=weights)

    return particles

def ProbModel(particles, model, time, sigma):
    # Calculate the proability of it begin from a particular direction
    mu = model(time)
    probs = np.zeros(particles.size)
    for i in range(particles.size):
        # Again, the observation model is normally distributed
        probs[i] = norm.pdf(particles[i], mu, sigma)

    return np.average(probs)


# Create Models

directions = ["left", "right", "back", "front"]

# Generate space for the model and their respective standard deviations
# Models are saved by saving the coefficients of the polynomial
leftModel = np.zeros(deg + 1)
leftSD = 0
leftCounter = 0
rightModel = np.zeros(deg + 1)
rightSD = 0
rightCounter = 0
frontModel = np.zeros(deg + 1)
frontSD = 0
frontCounter = 0
backModel = np.zeros(deg + 1)
backSD = 0
backCounter = 0

for root, folders, files in os.walk("../wav/Train"):
    # Traverse through the training set
        for name in files:
            direction = root[13:]
            # load the data
            samplerate, data = io.wavfile.read(os.path.join(root, name))
            length = data.shape[0] / samplerate

            # Set the time to go up to 5 seconds for polynomial fitting reasons
            time = np.linspace(0.,5., 5*samplerate)

            # we fill the excess time with zeros to help with the model fit 
            fill = np.zeros(time.size - len(data[:]))
            y = np.concatenate((data, fill))

            tempConst = np.polyfit(time, abs(y), deg)
            # store the data to the correct model
            if (direction == "left"):
                leftModel = (leftModel * leftCounter + tempConst)/(leftCounter + 1)
                leftSD = (leftSD * leftCounter +
                        np.std(abs(data[:])))/(leftCounter + 1)
                leftCounter = leftCounter + 1
            elif (direction == "right"):
                rightModel = (rightModel * rightCounter + tempConst)/(rightCounter + 1)
                rightSD = (rightSD * rightCounter +
                        np.std(abs(data[:])))/(rightCounter + 1)
                rightCounter = rightCounter + 1
            elif (direction == "front"):
                frontModel = (frontModel * frontCounter + tempConst)/(frontCounter + 1)
                frontSD = (frontSD * frontCounter +
                        np.std(abs(data[:])))/(frontCounter + 1)
                frontCounter = frontCounter + 1
            elif (direction == "back"):
                backModel = (backModel * backCounter + tempConst)/(backCounter + 1)
                backSD = (backSD * backCounter +
                        np.std(abs(data[:])))/(backCounter + 1)
                backCounter = backCounter + 1

# Create the acutal polynomial model
LeftModel = np.poly1d(leftModel)
RightModel = np.poly1d(rightModel)
FrontModel = np.poly1d(frontModel)
BackModel = np.poly1d(backModel)

# Initiaite particles for each direction
LeftParticles = np.zeros(numParticles)
RightParticles = np.zeros(numParticles)
FrontParticles = np.zeros(numParticles)
BackParticles = np.zeros(numParticles)

# run the test
# in the future, should be separating
for root, dirs, files in os.walk("../wav/Test"):
    for name in files:
            samplerate, testData = io.wavfile.read(os.path.join(root, name))
            length = testData.shape[0] / samplerate
            time = np.linspace(0., length, testData.shape[0])
            testModel = np.poly1d(np.polyfit(time, abs(testData[:]), deg))

            # Number of times we're going to be running the model
            numTests = int(length*freq)

            # Set up
            x = np.linspace(0., length, numTests)
            deltaTime = length/numTests


            # graph lines
            leftLine = np.zeros(x.size)
            rightLine = np.zeros(x.size)
            frontLine = np.zeros(x.size)
            backLine = np.zeros(x.size)

            # Run the particle filter loop
            for index, curTime in enumerate(x):

                # Run prediction for all the particles with their respective
                # models
                LeftPred = Predict(curTime, LeftParticles,
                        np.polyder(LeftModel), deltaTime, leftSD)
                RightPred = Predict(curTime, RightParticles, np.polyder(RightModel),
                        deltaTime, rightSD)
                FrontPred = Predict(curTime, FrontParticles, np.polyder(FrontModel),
                        deltaTime, frontSD)
                BackPred = Predict(curTime, BackParticles, np.polyder(BackModel),
                        deltaTime, backSD)

                # Do importance sampling
                LeftParticles = Update(LeftPred, testModel(curTime), leftSD)
                RightParticles = Update(RightPred, testModel(curTime), rightSD)
                FrontParticles = Update(FrontPred, testModel(curTime), frontSD)
                BackParticles = Update(BackPred, testModel(curTime), backSD)

                # Calculate the probabilities of seeing these particles
                LeftProb = ProbModel(LeftParticles, testModel, curTime, leftSD)
                RightProb = ProbModel(RightParticles, testModel, curTime, rightSD)
                FrontProb = ProbModel(FrontParticles, testModel, curTime, frontSD)
                BackProb = ProbModel(BackParticles, testModel, curTime, backSD)

                # Store them and continue
                leftLine[index] = LeftProb
                rightLine[index] = RightProb
                frontLine[index] = FrontProb
                backLine[index] = BackProb

            plt.plot(x, leftLine, 'r--', x, rightLine, 'bs', x, frontLine, 'g^',
                    x, backLine, '+')
            plt.legend(["Left", "Right", "Front", "Behind"])
            plt.xlabel("Time Step")
            plt.ylabel("Probability")
            filename = graphsFolder + "/probs-"
            title = ""
            if ("back" in name):
                title += "Behind - "
                filename += "behind_"
            elif ("front" in name):
                title += "Front - "
                filename += "front_"

            if ("left" in name):
                title += "Left"
                filename += "left-"
            elif ("right" in name):
                title += "Right"
                filename += "right-"

            title += ", Degree: " + str(deg)
            filename += "deg-" + str(deg)
            
            plt.title(title)
            plt.savefig(filename + ".png", format="png")
            plt.clf()

