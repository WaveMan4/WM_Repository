#pragma once
#include "bill.h"
#include "coin.h"
class ChangeReturnSlot
{
	public: 
		ChangeReturnSlot() { }
		Coin* returnedCoins;
		Bill* returnedBills;
		double changeInTotal = 0;
		void receiveCoins(Coin coins[]) { returnedCoins = coins; }
		void receiveBills(Bill bills[]) { returnedBills = bills; }
};