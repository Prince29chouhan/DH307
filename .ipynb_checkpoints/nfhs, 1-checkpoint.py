import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the Excel file
excel_file = 'NFHS_5_India_Districts_Factsheet_Data-1.xlsx'
df = pd.read_excel(excel_file)

#Replace *  values and other non-numerical values with NaN
df.replace(['*', '(', ')', '-'], '', regex=True, inplace=True)

# List of columns to convert to numeric, excluding non-numeric columns like District Names, State/UT
cols_to_numeric = df.columns.drop(['District Names', 'State/UT'])

# Convert columns to numeric, coercing errors to NaN
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values using the mean of each column
df.fillna(df.mean(), inplace=True)

# Identify columns for normalization (example - can be adjusted)
cols_to_normalize = [
    'Number of Households surveyed',
    'Number of Women age 15-49 years interviewed',
    'Number of Men age 15-54 years interviewed',
    'Female population age 6 years and above who ever attended school (%)',
    'Population below age 15 years (%)',
    ' Sex ratio of the total population (females per 1,000 males)',
    'Sex ratio at birth for children born in the last five years (females per 1,000 males)',
    'Children under age 5 years whose birth was registered with the civil authority (%)',
    'Deaths in the last 3 years registered with the civil authority (%)',
    'Population living in households with electricity (%)',
    'Population living in households with an improved drinking-water source1 (%)',
    'Population living in households that use an improved sanitation facility2 (%)',
    'Households using clean fuel for cooking3 (%)',
    'Households using iodized salt (%)',
    'Households with any usual member covered under a health insurance/financing scheme (%)',
    'Children age 5 years who attended pre-primary school during the school year 2019-20 (%)',
    'Women (age 15-49) who are literate4 (%)',
    'Women (age 15-49)  with 10 or more years of schooling (%)',
    'Women age 20-24 years married before age 18 years (%)',
    'Births in the 5 years preceding the survey that are third or higher order (%)',
    'Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
    'Women age 15-24 years who use hygienic methods of protection during their menstrual period5 (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Any method6 (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Any modern method6 (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Female sterilization (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Male sterilization (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - IUD/PPIUD (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Pill (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Condom (%)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Injectables (%)',
    'Total Unmet need for Family Planning (Currently Married Women Age 15-49  years)7 (%)',
    'Unmet need for spacing (Currently Married Women Age 15-49  years)7 (%)',
    'Health worker ever talked to female non-users about family planning (%)',
    'Current users ever told about side effects of current method of family planning8 (%)',
    'Mothers who had an antenatal check-up in the first trimester  (for last birth in the 5 years before the survey) (%)',
    'Mothers who had at least 4 antenatal care visits  (for last birth in the 5 years before the survey) (%)',
    'Mothers whose last birth was protected against neonatal tetanus (for last birth in the 5 years before the survey)9 (%)',
    'Mothers who consumed iron folic acid for 100 days or more when they were pregnant (for last birth in the 5 years before the survey) (%)',
    'Mothers who consumed iron folic acid for 180 days or more when they were pregnant (for last birth in the 5 years before the survey} (%)',
    'Registered pregnancies for which the mother received a Mother and Child Protection (MCP) card (for last birth in the 5 years before the survey) (%)',
    'Mothers who received postnatal care from a doctor/nurse/LHV/ANM/midwife/other health personnel within 2 days of delivery (for last birth in the 5 years before the survey) (%)',
    'Average out-of-pocket expenditure per delivery in a public health facility (for last birth in the 5 years before the survey) (Rs.)',
    'Children born at home who were taken to a health facility for a check-up within 24 hours of birth (for last birth in the 5 years before the survey} (%)',
    'Children who received postnatal care from a doctor/nurse/LHV/ANM/midwife/ other health personnel within 2 days of delivery (for last birth in the 5 years before the survey) (%)',
    'Institutional births (in the 5 years before the survey) (%)',
    'Institutional births in public facility (in the 5 years before the survey) (%)',
    'Home births that were conducted by skilled health personnel  (in the 5 years before the survey)10 (%)',
    'Births attended by skilled health personnel (in the 5 years before the survey)10 (%)',
    'Births delivered by caesarean section (in the 5 years before the survey) (%)',
    'Births in a private health facility that were delivered by caesarean section (in the 5 years before the survey) (%)',
    'Births in a public health facility that were delivered by caesarean section (in the 5 years before the survey) (%)',
    'Children age 12-23 months fully vaccinated based on information from either vaccination card or mother\'s recall11 (%)',
    'Children age 12-23 months fully vaccinated based on information from vaccination card only12 (%)',
    'Children age 12-23 months who have received BCG (%)',
    'Children age 12-23 months who have received 3 doses of polio vaccine13 (%)',
    'Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)',
    'Children age 12-23 months who have received the first dose of measles-containing vaccine (MCV) (%)',
    'Children age 24-35 months who have received a second dose of measles-containing vaccine (MCV) (%)',
    'Children age 12-23 months who have received 3 doses of rotavirus vaccine14 (%)',
    'Children age 12-23 months who have received 3 doses of penta or hepatitis B vaccine (%)',
    'Children age 9-35 months who received a vitamin A dose in the last 6 months (%)',
    'Children age 12-23 months who received most of their vaccinations in a public health facility (%)',
    'Children age 12-23 months who received most of their vaccinations in a private health facility (%)',
    'Prevalence of diarrhoea in the 2 weeks preceding the survey (Children under age 5 years) (%)',
    'Children with diarrhoea in the 2 weeks preceding the survey who received oral rehydration salts (ORS) (Children under age 5 years) (%)',
    'Children with diarrhoea in the 2 weeks preceding the survey who received zinc (Children under age 5 years) (%)',
    'Children swith diarrhoea in the 2 weeks preceding the survey taken to a health facility or health provider (Children under age 5 years) (%)',
    'Children Prevalence of symptoms of acute respiratory infection (ARI) in the 2 weeks preceding the survey (Children under age 5 years) (%)',
    'Children with fever or symptoms of ARI in the 2 weeks preceding the survey taken to a health facility or health provider (Children under age 5 years) (%)',
    'Children under age 3 years breastfed within one hour of birth15 (%)',
    'Children under age 6 months exclusively breastfed16 (%)',
    'Children age 6-8 months receiving solid or semi-solid food and breastmilk16 (%)',
    'Breastfeeding children age 6-23 months receiving an adequate diet16, 17  (%)',
    'Non-breastfeeding children age 6-23 months receiving an adequate diet16, 17 (%)',
    'Total children age 6-23 months receiving an adequate diet16, 17  (%)',
    'Children under 5 years who are stunted (height-for-age)18 (%)',
    'Children under 5 years who are wasted (weight-for-height)18 (%)',
    'Children under 5 years who are severely wasted (weight-for-height)19 (%)',
    'Children under 5 years who are underweight (weight-for-age)18 (%)',
    'Children under 5 years who are overweight (weight-for-height)20 (%)',
    'Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)',
    'Women (age 15-49 years) who are overweight or obese (BMI â‰¥25.0 kg/m2)21 (%)',
    'Women (age 15-49 years) who have high risk waist-to-hip ratio (â‰¥0.85) (%)',
    'Children age 6-59 months who are anaemic (<11.0 g/dl)22 (%)',
    'Non-pregnant women age 15-49 years who are anaemic (<12.0 g/dl)22 (%)',
    'Pregnant women age 15-49 years who are anaemic (<11.0 g/dl)22 (%)',
    'All women age 15-49 years who are anaemic22 (%)',
    'All women age 15-19 years who are anaemic22 (%)',
    'Women  age 15 years and above with high (141-160 mg/dl) Blood sugar level23 (%)',
    'Women age 15 years and above wih very high (>160 mg/dl) Blood sugar level23 (%)',
    'Women age 15 years and above wih high or very high (>140 mg/dl) Blood sugar level or taking medicine to control blood sugar level23 (%)',
    'Men age 15 years and above wih high (141-160 mg/dl) Blood sugar level23 (%)',
    'Men (age 15 years and above wih  very high (>160 mg/dl) Blood sugar level23 (%)',
    'Men age 15 years and above wih high or very high (>140 mg/dl) Blood sugar level  or taking medicine to control... blood sugar level23 (%)',
    'Women age 15 years and above wih Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)',
    'Women age 15 years and above wih Moderately or severely elevated blood pressure (Systolic â‰¥160 mm of Hg and/or Diastolic â‰¥100 mm of Hg) (%)',
    'Women age 15 years and above wih Elevated blood pressure (Systolic â‰¥140 mm of Hg and/or Diastolic â‰¥90 mm of Hg) or taking medicine to control blood pressure (%)',
    'Men age 15 years and above wih Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)',
    'Men age 15 years and above wih Moderately or severely elevated blood pressure (Systolic â‰¥160 mm of Hg and/or Diastolic â‰¥100 mm of Hg) (%)',
    'Men age 15 years and above wih Elevated blood pressure (Systolic â‰¥140 mm of Hg and/or Diastolic â‰¥90 mm of Hg) or taking medicine to control blood pressure (%)',
    'Women (age 30-49 years) Ever undergone a screening test for cervical cancer (%)',
    'Women (age 30-49 years) Ever undergone a breast examination for breast cancer (%)',
    'Women (age 30-49 years) Ever undergone an oral cavity examination for oral cancer (%)',
    'Women age 15 years and above who use any kind of tobacco (%)',
    'Men age 15 years and above who use any kind of tobacco (%)',
    'Women age 15 years and above who consume alcohol (%)',
    'Men age 15 years and above who consume alcohol (%)'
]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Display the first few rows of the normalized DataFrame
print(df.head())

# Example of normalizing *by* a proxy for population (Household Survey Size):

df['Mortality Rate Normalized by Households'] = df['Deaths in the last 3 years registered with the civil authority (%)'] / df['Number of Households surveyed']

print(df[['District Names', 'State/UT', 'Number of Households surveyed', 'Deaths in the last 3 years registered with the civil authority (%)','Mortality Rate Normalized by Households']].head())
