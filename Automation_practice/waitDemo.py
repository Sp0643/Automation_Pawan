#Implicit wait  -
#Explicit Wait
import time

from selenium import webdriver
#pause the test for few seconds using Time class
driver = webdriver.Chrome(executable_path="C:\\chromedriver.exe")
driver.implicitly_wait(5)
# wait until 5 seconds if object is not displayed
#Global wait
#1.5 second to reach next page- execution will resume in 1.5 seconds
#if object do not show up at all, then max time your test waits for 5 seconds
driver.get("https://rahulshettyacademy.com/seleniumPractise/")
driver.find_element_by_css_selector("input.search-keyword").send_keys("ber")
time.sleep(4)
count =len(driver.find_elements_by_xpath("//div[@class='products']/div"))
assert count == 3
buttons = driver.find_elements_by_xpath("//div[@class='product-action']/button")

for button in buttons:
    button.click()

driver.find_element_by_css_selector("img[alt='Cart']").click()
driver.find_element_by_xpath("//button[text()='PROCEED TO CHECKOUT']").click()
driver.find_element_by_class_name("promoCode").send_keys("rahulshettyacademy")
driver.find_element_by_css_selector(".promoBtn").click()
print(driver.find_element_by_css_selector("span.promoInfo").text




Certainly! Here's an example code snippet using the `shareplum` library to connect to SharePoint and perform basic actions like listing files in a document library:

```python
from shareplum import Site
from shareplum import Office365

# SharePoint site and authentication details
site_url = "https://your-sharepoint-site-url/sites/your-site"
username = "your-username"
password = "your-password"

# Connect to SharePoint site
authcookie = Office365(site_url, username=username, password=password).GetCookies()
site = Site(site_url, authcookie=authcookie)

# Specify document library name
library_name = "Shared Documents"

# Get the document library
doc_library = site.List(library_name)

# List all files in the document library
files = doc_library.GetListItems(viewname="All Documents")
for file in files:
    print(file["FileLeafRef"])

# Close the connection
site._session.close()
```

Make sure to replace `https://your-sharepoint-site-url/sites/your-site`, `your-username`, and `your-password` with your actual SharePoint site URL, username, and password. Additionally, you may need to install the `shareplum` library using `pip install shareplum` before running the code.

This example demonstrates how to connect to a SharePoint site, authenticate, and list files in a document library. You can further extend the code to perform other actions like uploading files, creating folders, updating items, etc., using the `shareplum` library.







connecting with SharePoint using selenium


Certainly! Since you're using Selenium to interact with SharePoint and want to create a SharePoint list using the data from a DataFrame, I'll provide you with a code example to achieve this. Here's a step-by-step guide and sample code:

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time

# Create a DataFrame with sample data
data = {
    'Serial Number': [1, 2, 3],
    'Name': ['John', 'Jane', 'Michael'],
    'Employee Name': ['John Doe', 'Jane Smith', 'Michael Johnson'],
    'KAD': ['12345', '67890', '54321']
}
df = pd.DataFrame(data)

# Start Selenium WebDriver (adjust the path to your ChromeDriver executable)
driver = webdriver.Chrome(executable_path='path/to/chromedriver')

# Navigate to the SharePoint list creation page
driver.get('https://your-sharepoint-site-url/_layouts/15/start.aspx#/Lists/')

# Wait for the page to load
time.sleep(5)

# Find and click the "New" button to create a new item
new_button = driver.find_element_by_id('idHomePageNewItem')
new_button.click()

# Wait for the new item form to load
time.sleep(3)

# Fill in the form fields using DataFrame data
serial_number_field = driver.find_element_by_id('YourSerialNumberFieldId')
serial_number_field.send_keys(str(df.loc[0, 'Serial Number']))

name_field = driver.find_element_by_id('YourNameFieldId')
name_field.send_keys(df.loc[0, 'Name'])

employee_name_field = driver.find_element_by_id('YourEmployeeNameFieldId')
employee_name_field.send_keys(df.loc[0, 'Employee Name'])

kad_field = driver.find_element_by_id('YourKADFieldId')
kad_field.send_keys(df.loc[0, 'KAD'])

# Submit the form
submit_button = driver.find_element_by_id('YourSubmitButtonId')
submit_button.click()

# Wait for the item to be added
time.sleep(3)

# Repeat the above steps for each row in the DataFrame

# Close the WebDriver
driver.quit()
```

In this code, replace `'path/to/chromedriver'` with the actual path to your ChromeDriver executable, and replace `'https://your-sharepoint-site-url'` with your SharePoint site URL. Adjust the field IDs and element locators (`find_element_by_id`) according to your SharePoint list form.

Please note that using Selenium to interact with SharePoint involves navigating the user interface, filling out forms, and clicking buttons, which can be slower and less reliable than using direct API methods. If you can resolve the SSL error, it would be more efficient and robust to use REST API for SharePoint list operations.