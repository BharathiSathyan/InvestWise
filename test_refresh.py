from risklens.data import refresh_market_data

prices, status = refresh_market_data(force_refresh=True)

print("\nRefresh status:")
print(status)

print("\nDataset shape:")
print(prices.shape)

print("\nLatest available observation:")
print(prices.tail(1))

print("\nLatest date:")
print(prices.index.max())