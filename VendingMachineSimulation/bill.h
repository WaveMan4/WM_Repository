#pragma once

#include <string>
using namespace std;
class Bill
{
	private: 
		double billValue;
		string type;
	public:
		Bill() { billValue = 0; type = "default";  }
		Bill(double val, string bill) { billValue = val; type = bill; }
		double getBillValue() { return billValue; }
		void setBillValue(double value) { billValue = value; }
		string getType() { return type; }
		void setType(string t) { type = t; }
};