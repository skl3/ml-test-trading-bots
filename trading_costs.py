'''
Incorporate trading costs such as commision and factoring in slippage
'''
SLIPPAGE = 0.6
STOP = -3
trades = testing[testing['predictions'] == 1][('datetime', 'gain', 'ho', 'lo', 'open', 'close')]
trades['pnl'] = trades.apply(lambda x: trade_with_stop(x, slippage=SLIPPAGE, stop=STOP))
plot_equity_chart(trades['pnl'],'Decision tree model')
print("Slippage is %s, STOP level at %s" % (SLIPPAGE, STOP))