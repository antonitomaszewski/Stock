import bollinger as B

stock = B.bollinger(time_frame=10_000, n=20, time_step=1)
print(stock)
window = B.Window(stock.transactions[100])
window.data.iloc[-1].price
window.normalize()
window.draw()
