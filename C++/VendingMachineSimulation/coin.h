#pragma once
#include <string>

using namespace std;
class Coin
{
	private: 
		double coinValue;
		string type;
	public:
		Coin() { coinValue = 0; type = "default"; }
		Coin(double val, string coin) { coinValue = val; type = coin; }
		double getCoinValue() { return coinValue; }
		void setCoinValue(double value) { coinValue = value; }
		string getType() { return type; }
		void setType(string t) { type = t; }
};