# This is a helper function to trade 1 bar (for example 1 day) with a Buy order at opening session
# and a Sell order at closing session. To protect against adverse movements of the price, a STOP order
# will limit the loss to the stop level (stop parameter must be a negative number)
# each bar must contains the following attributes: 
# Open, High, Low, Close prices as well as gain = Close - Open and lo = Low - Open
def trade_with_stop(bar, slippage = 0, stop=None):
    """
    Given a bar, with a gain obtained by the closing price - opening price
    it applies a stop limit order to limit a negative loss
    If stop is equal to None, then it returns bar['gain']
    """
    bar['gain'] = bar['gain'] - slippage
    if stop is None:
        real_stop = stop - slippage
        if bar['lo']<=stop:
            return real_stop
    # stop == None    
    return bar['gain']