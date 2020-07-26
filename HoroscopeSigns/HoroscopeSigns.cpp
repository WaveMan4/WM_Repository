// HoroscopeSigns.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

using namespace std;
const int console_width = 80;

int main()
{
    map<int, int> ascendant_days;
    ascendant_days[1] = 21;
    ascendant_days[2] = 20;
    ascendant_days[3] = 20;
    ascendant_days[4] = 19;
    ascendant_days[5] = 20;
    ascendant_days[6] = 21;
    ascendant_days[7] = 23;
    ascendant_days[8] = 22; 
    ascendant_days[9] = 23;
    ascendant_days[10] = 23;
    ascendant_days[11] = 23;
    ascendant_days[12] = 21;

    map<string, string> sign_attributes;
    sign_attributes["Aries"] = "Initiator, Fiery, and Confident";
    sign_attributes["Taurus"] = "Reliable, Stubborn, and Materialistic";
    sign_attributes["Gemini"] = "Sociable, Two-Faced, and Witty";
    sign_attributes["Cancer"] = "Protective, Moody, and Compassionate";
    sign_attributes["Leo"] = "Passionate, Arrogant, and Competitive";
    sign_attributes["Virgo"] = "Thorough, Over-Thinking, and Perfectionist";
    sign_attributes["Libra"] = "Diplomatic, Indecisive, and Charming";
    sign_attributes["Scorpio"] = "Ambitious, Resentful, and Intense";
    sign_attributes["Sagittarius"] = "Adventurous, Selfish, and Optimistic";
    sign_attributes["Capricorn"] = "Hard-working, Pessimistic, and Practical";
    sign_attributes["Aquarius"] = "Assertive, Uncompromising, and Rebellious";
    sign_attributes["Pisces"] = "Impressionable, Self-Undoing, and Mystical";

    int month, day, year;

    string sign;

    cout << "*****************************************" << endl;
    cout << "==> WELCOME TO THE FORTUNE TELLER APP <==" << endl;
    cout << "*****************************************" << endl;
    cout << "\n\n\n";
 
    cout << "\tPlease type your birthday below: \n";
    cout << "\tMonth (type number between 1 and 12)\n";
    cout << "\tDay (type number between 1 and 31)\n";
    cout << "\tYear (type a four-digit number)\n";
    cout << setw(console_width / 2 - 16) << right << "Month: "; cin >> month;
    cout << setw(console_width / 2 - 16) << right << "Day: "; cin >> day;
    cout << setw(console_width / 2 - 16) << right << "Year: "; cin >> year;

    switch (month) {
        case 1: 
            sign = day < ascendant_days[month] ? "Capricorn" : "Aquarius";
            break;
        case 2:
            sign = day < ascendant_days[month] ? "Aquarius" : "Pisces";
            break;
        case 3:
            sign = day < ascendant_days[month] ? "Pisces" : "Aries";
            break;
        case 4:
            sign = day < ascendant_days[month] ? "Aries" : "Taurus";
            break;
        case 5:
            sign = day < ascendant_days[month] ? "Taurus" : "Gemini";
            break;
        case 6:
            sign = day < ascendant_days[month] ? "Gemini" : "Cancer";
            break;
        case 7:
            sign = day < ascendant_days[month] ? "Cancer" : "Leo";
            break;
        case 8:
            sign = day < ascendant_days[month] ? "Leo" : "Virgo";
            break;
        case 9:
            sign = day < ascendant_days[month] ? "Virgo" : "Libra";
            break;
        case 10:
            sign = day < ascendant_days[month] ? "Libra" : "Scorpio";
            break;
        case 11:
            sign = day < ascendant_days[month] ? "Scorpio" : "Sagittarius";
            break;
        case 12:
            sign = day < ascendant_days[month] ? "Sagittarius" : "Capricorn";
            break;
        default: 
            break;
    }
    
    cout << "\n\n";

    string horoscope_fp = sign + ".txt";
    ifstream hfile(horoscope_fp);
    string line; 

    cout << "The Zodiac sign at this birthday is:\t" << sign << endl;
    cout << "The traits of this zodiac sign are:\t" << sign_attributes[sign] << endl << endl;;

    if (hfile.is_open())
    {
        while (getline(hfile, line))
        {
            cout << "\t\t" << line << endl;
        }
        hfile.close();
        cout << endl << endl << endl;
    }
    else cout << "Error: File was not found" << endl;

    return 0;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
