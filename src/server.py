from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from os import getcwd, path

print(getcwd())
# filePath = path.join(getcwd(), 'src', 'pima-indians-diabetes.csv')
# dataset = loadtxt(filePath, delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:,0:8]
# y = dataset[:,8]

X = np.array([])


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

result, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
print(result)

predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))