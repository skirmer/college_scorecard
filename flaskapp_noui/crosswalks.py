
# Values
reg = [('U.S. Service Schools', '0'),
('New England (CT, ME, MA, NH, RI, VT)','1'),
('Mid East (DE, DC, MD, NJ, NY, PA)', '2'),
('Great Lakes (IL, IN, MI, OH, WI)', '3'),
('Plains (IA, KS, MN, MO, NE, ND, SD)', '4'),
('Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)', '5'),
('Southwest (AZ, NM, OK, TX)', '6'),
('Rocky Mountains (CO, ID, MT, UT, WY)', '7'),
('Far West (AK, CA, HI, NV, OR, WA)', '8'),
('Outlying Areas (AS, FM, GU, MH, MP, PR, PW, VI)', '9')]

coltype = ['Public',
'Private, nonprofit',
'Private, for-profit']

locs = [
('City: Large (population of 250,000 or more)', '11')
,('City: Midsize (population of at least 100,000 but less than 250,000)', '12')
,('City: Small (population less than 100,000)', '13')
,('Suburb: Large (outside principal city, in urbanized area with population of 250,000 or more)', '21')
,('Suburb: Midsize (outside principal city, in urbanized area with population of at least 100,000 but less than 250,000)', '22')
,('Suburb: Small (outside principal city, in urbanized area with population less than 100,000)', '23')
,('Town: Fringe (in urban cluster up to 10 miles from an urbanized area)', '31')
,('Town: Distant (in urban cluster more than 10 miles and up to 35 miles from an urbanized area)', '32')
,('Town: Remote (in urban cluster more than 35 miles from an urbanized area)', '33')
,('Rural: Fringe (rural territory up to 5 miles from an urbanized area or up to 2.5 miles from an urban cluster)', '41')
,('Rural: Distant (rural territory more than 5 miles but up to 25 miles from an urbanized area or more than 2.5 and up to 10 miles from an urban cluster)', '42')
,('Rural: Remote (rural territory more than 25 miles from an urbanized area and more than 10 miles from an urban cluster)', '43')
]

cipcats = [
('AGRICULTURE, AGRICULTURE OPERATIONS, AND RELATED SCIENCES', '01'),
 ('NATURAL RESOURCES AND CONSERVATION', '03'),
 ('ARCHITECTURE AND RELATED SERVICES', '04'),
 ('AREA, ETHNIC, CULTURAL, AND GENDER STUDIES', '05'),
 ('COMMUNICATION, JOURNALISM, AND RELATED PROGRAMS', '09'),
 ('COMMUNICATIONS TECHNOLOGIES/TECHNICIANS AND SUPPORT SERVICES', '10'),
 ('COMPUTER AND INFORMATION SCIENCES AND SUPPORT SERVICES', '11'),
 ('PERSONAL AND CULINARY SERVICES', '12'),
 ('EDUCATION', '13'),
 ('ENGINEERING', '14'),
 ('ENGINEERING TECHNOLOGIES/TECHNICIANS', '15'),
 ('FOREIGN LANGUAGES, LITERATURES, AND LINGUISTICS', '16'),
 ('FAMILY AND CONSUMER SCIENCES/HUMAN SCIENCES', '19'),
 ('LEGAL PROFESSIONS AND STUDIES', '22'),
 ('ENGLISH LANGUAGE AND LITERATURE/LETTERS', '23'),
 ('LIBERAL ARTS AND SCIENCES, GENERAL STUDIES AND HUMANITIES', '24'),
 ('LIBRARY SCIENCE', '25'),
 ('BIOLOGICAL AND BIOMEDICAL SCIENCES', '26'),
 ('MATHEMATICS AND STATISTICS', '27'),
 ('RESERVE OFFICER TRAINING CORPS (JROTC, ROTC', '28'),
 ('MILITARY TECHNOLOGIES', '29'),
 ('MULTI/INTERDISCIPLINARY STUDIES', '30'),
 ('PARKS, RECREATION, LEISURE, AND FITNESS STUDIES', '31'),
 ('BASIC SKILLS', '32'),
 ('CITIZENSHIP ACTIVITIES', '33'),
 ('HEALTH-RELATED KNOWLEDGE AND SKILLS', '34'),
 ('INTERPERSONAL AND SOCIAL SKILLS', '35'),
 ('LEISURE AND RECREATIONAL ACTIVITIES', '36'),
 ('PERSONAL AWARENESS AND SELF-IMPROVEMENT', '37'),
 ('PHILOSOPHY AND RELIGIOUS STUDIES', '38'),
 ('THEOLOGY AND RELIGIOUS VOCATIONS', '39'),
 ('PHYSICAL SCIENCES', '40'),
 ('SCIENCE TECHNOLOGIES/TECHNICIANS', '41'),
 ('PSYCHOLOGY', '42'),
 ('SECURITY AND PROTECTIVE SERVICES', '43'),
 ('PUBLIC ADMINISTRATION AND SOCIAL SERVICE PROFESSIONS', '44'),
 ('SOCIAL SCIENCES', '45'),
 ('CONSTRUCTION TRADES', '46'),
 ('MECHANIC AND REPAIR TECHNOLOGIES/TECHNICIANS', '47'),
 ('PRECISION PRODUCTION', '48'),
 ('TRANSPORTATION AND MATERIALS MOVING', '49'),
 ('VISUAL AND PERFORMING ARTS', '50'),
 ('HEALTH PROFESSIONS AND RELATED CLINICAL SCIENCES', '51'),
 ('BUSINESS, MANAGEMENT, MARKETING, AND RELATED SUPPORT SERVICES', '52'),
 ('HIGH SCHOOL/SECONDARY DIPLOMAS AND CERTIFICATES', '53'),
 ('HISTORY', '54'),
 ('Residency Programs', '60')]