#pragma once
#include "bill.h"
#include "coin.h"
class CoinBillSlot
{
	public:
	CoinBillSlot() { currentTotal = 0; insertedBills = new Bill(); insertedCoins = new Coin(); }
	Coin* insertedCoins;
	Bill* insertedBills;
	double currentTotal;

	void clearSlot() { insertedCoins = NULL; insertedBills = NULL; }
	void insertCoins(Coin coinArray[]) 
	{ 
		insertedCoins = coinArray;
	}
	void insertBills(Bill billArray[]) 
	{ 
		insertedBills = billArray;
	}
	Bill* getBills() { return insertedBills; }
	Coin* getCoins() { return insertedCoins; }
	double getCurrentTotal() { return currentTotal; }
	void setCurrentTotal(double val) { currentTotal = val; }
};