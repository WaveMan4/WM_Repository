// MiniProject1_gkepnang.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#include <iomanip> 

#include "bill.h"
#include "bin.h"
#include "button.h"
#include "changereturnslot.h"
#include "coin.h"
#include "coinbillslot.h"
#include "deliveryslot.h"
#include "item.h"

//provide namespace for std
using namespace std;

/*
	includes relevant header files:
	1) Bill
	2) Bin
	3) Button
	4) ChangeReturnSlot
	5) Coin
	6) CoinBillSlot
	7) DeliverySlot
	8) Item
*/

/*
	Create Vending Machine class here
*/
class VendingMachine
{
	private: 
		Bin binItems[6][4];
		int row = 6;
		int column = 4;
		char types[6][15] = { "Chips", "Chocolate", "Fruit Snack", "Breakfast", "Cookie", "Chewing Gum" };
		char chips[4][15] = { "Lays", "Doritos", "Rold Gold",  "Cheetos" };
		char chocolates[4][15] = { "Snickers", "Mars", "Hersheys",  "M&M's" };
		char fruit_snacks[4][15] = { "Welch's", "Gummies", "Gushers",  "Skittles" };
		char breakfasts[4][15] = { "Pop Tarts", "Kudos", "Quakers",  "Nature Valley" };
		char cookies[4][15] = { "Famous Amos", "Oreo", "Keebler",  "Chips Ahoy" };
		char chewing_gums[4][15] = { "Winterfresh", "Big Red", "Spearmint",  "Doublemint" };
		double chipPrices[4] = { 1.50, 1.15, 1.10, 1.25 };
		double chocolatePrices[4] = { 1.25, 1.25, 1.40, 1.30 };
		double fruitsnackPrices[4] = { 1.60, 1.35, 1.30, 1.40 };
		double breakfastPrices[4] = { 1.25, 1.40, 1.70, 1.90 };
		double cookiePrices[4] = { 1.50, 1.45, 1.50, 1.35 };
		double gumPrices[4] = { 1.15, 1.15, 1.15, 1.15 };
		DeliverySlot deliverySlot;
		CoinBillSlot coinbillSlot;
		Button selectionButtons[6][4];
		ChangeReturnSlot changeReturnSlot;

	public:	
		VendingMachine() 
		{
			createInventory(6, 4);
			deliverySlot = DeliverySlot();
			coinbillSlot = CoinBillSlot();
			changeReturnSlot = ChangeReturnSlot();
		}
		const int maxBills = 3;
		const int maxCoins = 40;
		CoinBillSlot getCoinBillSlot() { return this->coinbillSlot; }
		void setCoinBillSlot(CoinBillSlot cbs) { coinbillSlot = cbs; }
		DeliverySlot getDeliverySlot() { return this->deliverySlot; }
		ChangeReturnSlot getChangeReturnSlot() { return this->changeReturnSlot; }

		void createInventory(int row, int column) 
		{
			Item newItem;
			Bin newBin;
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					selectionButtons[i][j].setxcoord(j);
					selectionButtons[i][j].setycoord(i);
					if (i == 0)
					{
						newItem = Item(chips[j], types[i], chipPrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else if (i == 1)
					{
						newItem = Item(chocolates[j], types[i], chocolatePrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else if (i == 2)
					{
						newItem = Item(fruit_snacks[j], types[i], fruitsnackPrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else if (i == 3)
					{
						newItem = Item(breakfasts[j], types[i], breakfastPrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else if (i == 4)
					{
						newItem = Item(cookies[j], types[i], cookiePrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else if (i == 5)
					{
						newItem = Item(chewing_gums[j], types[i], gumPrices[j]);
						binItems[i][j] = Bin(newItem, j, i);
					}
					else continue;
				}
			}
		}
		void printInventory() {
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					binItems[i][j].printItemsFromBin();
				}
				cout << endl;
			}
		}
		bool processTransaction(int row, int column, double inputTotal)
		{
			Bin currentBin;
			Item currentItem;

			//first retrieve object at [row,column]
			if (!(binItems[row][column].binIsEmpty()))
			{
				currentBin = binItems[row][column];
				currentItem = currentBin.getItem();
			}
			else return false;

			//	check whether user specified enough money
				//-->if enough money, then:
				//	1) purchase the indicated/selected item
				//	2) supply the money to the vending machine
				//	3) return change for the monetary item
			double change = (inputTotal - currentItem.getPrice());
			double currentPrice = currentItem.getPrice();
			if (change >= 0)
			{
				deliverySlot.returnedItem = currentItem;
				//place change in change-return slot
				Bill billArray[3]; 
				int billIndex = 0;
				Coin coinArray[40];
				int coinIndex = 0;
				while (change > 0.001)
				{
					if (change > 5.00)
					{
						billArray[billIndex] = Bill(5, "$5");
						change -= 5.00;
						billIndex++;
					}
					else if (change > 1.00)
					{
						billArray[billIndex] = Bill(1, "$1");
						change -= 1.00;
						billIndex++;
					}
					else if (change > 0.50)
					{
						coinArray[coinIndex] = Coin(0.50, "half-dollar");
						change -= 0.50;
						coinIndex++;
					}
					else if (change > 0.25)
					{
						coinArray[coinIndex] = Coin(0.25, "quarter");
						change -= 0.25;
						coinIndex++;
					}
					else if (change > 0.10)
					{
						coinArray[coinIndex] = Coin(0.10, "dime");
						change -= 0.10;
						coinIndex++;
					}
					else if (change > 0.05)
					{
						coinArray[coinIndex] = Coin(0.05, "nickel");
						change -= 0.05;
						coinIndex++;
					}
				}
				this->changeReturnSlot.receiveBills(billArray);
				this->changeReturnSlot.receiveCoins(coinArray);

				//clear input slot
				this->coinbillSlot.clearSlot();

				//Provide transaction summary
				cout << "Item purchased: "; currentItem.printAttributes(currentBin.xcoord, currentBin.ycoord); 
				cout << endl;
				cout << "Change for Item listed below: " << endl;
				for (int i = 0; i < 3; i++)
				{
					if (billArray[i].getType() != "default")
						cout << billArray[i].getType() << endl;
				}
				for (int j = 0; j < 40; j++)
				{
					if (coinArray[j].getType() != "default")
						cout << coinArray[j].getType() << endl;
				}
				return true;
			}
				//-->if not enough money, then:
				//	1) abort transaction
				//	2) display indication to user that "Money was not accepted. Product was not purchased."
				//	3) prompt user for a) another transaction or b) exit program
			else
			{
				currentBin.setEmptyFlag(false);
				cout << "Money was not accepted. Product was not purchased.";
				return false;
			}
		}
};


//menu of commands is provided in "main" function
int main()
{
	// Initialize flag for active customer session
	bool customerAtVendingMachine = true;

	// Initialize input variables for user
	char userInput; 
	int inputRow;
	int inputColumn;
	double bill;
	char coin;

	//Initialize money total variable
	double inputTotal;

	//Welcome message for Vending Machine
    cout << "Welcome to Mini Project 1 - JHU Vending Machine!!\n\n";

	//Instantiate VendingMachine
	VendingMachine vendingMachine = VendingMachine();

	while(customerAtVendingMachine)
	{ 
		//Display list of commands
		cout << "Vending Machine Commands " << endl
			<< "\t 1) 'p' or 'P' - INITIALIZE PURCHASE, " << endl
			<< "\t 2) 'x' or 'X' - EXIT FROM TRANSACTION." << endl << endl;
		//Display the vending machine inventory
		vendingMachine.printInventory();
			//-->Prompt customer for command
			//-->Process command
		cout << "User please select a Vending Machine Command - either 'p' or 'x': ";
		cin >> userInput;
		if (userInput == 'x' || userInput == 'X') 
			customerAtVendingMachine = false;
		else if (userInput == 'p' || userInput == 'P')
		{
			//If purchase: 
			//	prompt user to indicate what item to purchase
			cout << "Now you can purchase an item.\nPlease provide Row and Column." << endl;
			cout << "Row: "; cin >> inputRow; 
			
			cout << "Column: "; cin >> inputColumn;
			if (inputRow != 0 && inputColumn != 0)
			{
				//	prompt the user to specify monetary items used for payment
				cout << "Now provide your money for item." << endl;

				cout << "Please input bills first. Press 1 or 5 for Dollars repeatedly..."
					<< "then press C to complete transaction" << endl;
				inputTotal = 0;
				bill = 99;
				Bill billArray[3];
				int index = 0;
 				while (index < 3)
				{
					cout << "Insert bill: ";
					cin >> bill;
					if (bill == 1 || bill == 5)
					{
						if (bill == 1)
						{
							billArray[index] = Bill(1, "$1");
							inputTotal += bill; index++;
						}
						else if (bill == 5)
						{
							billArray[index] = Bill(5, "$5");
							inputTotal += bill; index++;
						}
					}
					else
					{
						bill = 99;
						break;
					}
				}
				cin.clear();
				cin.ignore();
				CoinBillSlot cbs = vendingMachine.getCoinBillSlot(); 
				cbs.insertBills(billArray);
				vendingMachine.setCoinBillSlot(cbs);


				cout << "Please input coins first. Press H for half-dollar, Q for quarter, D for dime, N for nickel repeatedly..."
					<< "\n\t then press C to complete transaction" << endl;
				coin = 0;
				Coin coinArray[40];
				index = 0;
				while (index != 40)
				{
					cout << "Insert coin: ";
					cin >> coin;
					if (coin == 'c' || coin == 'C') break;
					if (coin == 'h' || coin == 'H') 
					{ 
						coinArray[index] = Coin(0.5, "half-dollar");
						inputTotal += 0.5; index++;
					}
					else if (coin == 'q' || coin == 'Q') 
					{ 
						coinArray[index] = Coin(0.25, "quarter");
						inputTotal += 0.25; index++;
					}
					else if (coin == 'd' || coin == 'D') 
					{ 
						coinArray[index] = Coin(0.1, "dime");
						inputTotal += 0.1; index++;
					}
					else if (coin == 'n' || coin == 'N') 
					{ 
						coinArray[index] = Coin(0.05, "nickel");
						inputTotal += 0.05; index++;
					}
					else 
					{
						cout << "Input is incorrect. User try again." << endl;
						break;
					}	
					coin = 0;
				}
				//coinptr = coinArray;
				cbs = vendingMachine.getCoinBillSlot();
				cbs.insertCoins(coinArray);
				vendingMachine.setCoinBillSlot(cbs);
				
				//Display the money currently held in machine
				cout << "Money currently in VendingMachine: \n\t";
				cbs = vendingMachine.getCoinBillSlot();
				cout << "$" << setprecision(2) << inputTotal << endl << endl;

				if ((inputRow >= 0 && inputRow <= 5) && (inputColumn >= 0 && inputColumn <= 5))
				{
					if (vendingMachine.processTransaction(inputRow, inputColumn, inputTotal))
						continue;
					else
						cout << "Transaction error. Please restart" << endl;
				}
				else break;
			}
			else
			{
				break;
			}
		}
		else
		{
			break;
		}
		cin.clear();
		cin.ignore();

		userInput = 0;
		while (userInput != 'c' && userInput != 'C')
		{
			cout << "Press 'c' or 'C' to continue." << endl;
			cin >> userInput;
		}

		//Clear terminal screen
		system("CLS");

		//return to top of display loop
		cin.clear();
		cin.ignore();
	}
	cout << endl << "Exiting program now...\tThank you for using JHU Vending Machine!!" << endl << endl;
	return 0;
}
