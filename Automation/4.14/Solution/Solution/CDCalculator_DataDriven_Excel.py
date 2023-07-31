import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from doubts import XLUtils

serv_obj=Service("C:\Drivers\chromedriver_win32\chromedriver.exe")
driver=webdriver.Chrome(service=serv_obj)
driver.implicitly_wait(10)

driver.get("https://www.cit.com/cit-bank/resources/calculators/certificate-of-deposit-calculator/")
driver.maximize_window()

inideposit = driver.find_element(By.XPATH,"//input[@id='mat-input-0']")
length = driver.find_element(By.XPATH,"//input[@id='mat-input-1']")
apr = driver.find_element(By.XPATH,"//input[@id='mat-input-2']")
calbutton = driver.find_element(By.XPATH,"//button[@id='CIT-chart-submit']")

path = "C:\\SeleniumPractice\\caldata2.xlsx"
rows = XLUtils.getRowCount(path, "Sheet1")
print("row count is : " , rows)

for r in range(2,rows+1):
    inidepo = XLUtils.readData(path, "Sheet1", r, 1)
    interestrate = XLUtils.readData(path, "Sheet1", r, 2)
    monthlength = XLUtils.readData(path, "Sheet1", r, 3)
    compoundingmonths = XLUtils.readData(path, "Sheet1", r, 4)
    exptotal = XLUtils.readData(path, "Sheet1", r, 5)

    inideposit.clear()
    length.clear()
    apr.clear()
    time.sleep(3)

    inideposit.send_keys(inidepo)
    length.send_keys(monthlength)
    apr.send_keys(interestrate)

    #Dropdown (Boostrap) - Not having Select Tag
    compoundrp = driver.find_element(By.XPATH,"//mat-select[@id='mat-select-0']")
    compoundrp.click()
    options = driver.find_elements(By.XPATH,"//div[@id='mat-select-0-panel']//mat-option")

    for option in options:
        if(option.text==compoundingmonths):
            option.click()

    calbutton.click()
    acttotal = driver.find_element(By.XPATH,"//span[@id='displayTotalValue']").text

    print("exp total is from excel: ", exptotal)
    print("act total is from app: ", acttotal)

    if(exptotal==acttotal):
        XLUtils.writeData(path, "Sheet1" ,r, 7 ,"Passed")
        XLUtils.fillGreenColor(path, "Sheet1" ,r, 7)
    else:
        XLUtils.writeData(path, "Sheet1" ,r, 7 ,"Failed")
        XLUtils.fillRedColor(path, "Sheet1" ,r, 7)

print("calculation has been completed")
driver.close()
