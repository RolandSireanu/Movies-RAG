from collections import Counter



print(Counter("ABCDEABCD"))
print(Counter({"A":1, "B":2}))
c = Counter(me=3, you=2)
print(c["him"])
print(c["me"])