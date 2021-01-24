import bollinger as B
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

stock = B.bollinger(time_frame=100_000, n=20, time_step=10)

stock
stock.get_data_for_ML()[1].sum()


clf = LogisticRegression(random_state=0).fit(X, y)
clf.score(X, y)
clf.predict(X)
neigh = KNeighborsClassifier(n_neighbors=3).fit(X, y)
neigh.score(X, y)
sum(neigh.predict(X)) / len(y)
for k in range(1, 20, 2):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    print(f"k = {k}")
    print(sum(neigh.predict(X)) / len(neigh.predict(X)))
