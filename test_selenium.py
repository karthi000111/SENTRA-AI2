import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_ui(pdf_path):
    print(f"\n==================================================")
    print(f"UI E2E TEST: {os.path.basename(pdf_path)}")
    print(f"==================================================")
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("http://localhost:8501")
        
        wait = WebDriverWait(driver, 10)
        
        # Click Create Research Session
        buttons = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "button")))
        for b in buttons:
            if "Create Research Session" in b.text:
                driver.execute_script("arguments[0].click();", b)
                break
                
        time.sleep(2)
        
        # Upload file (Streamlit input type file is hidden, but can still receive keys)
        file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        file_input.send_keys(os.path.abspath(pdf_path))
        
        time.sleep(2)
        
        # Click Index
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for b in buttons:
            if "Index uploaded documents" in b.text:
                driver.execute_script("arguments[0].click();", b)
                break
                
        time.sleep(15) # wait for FAISS indexing
        
        # Switch to Guardrail Tab
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for t in tabs:
            if "Implementation Guardrail" in t.text:
                driver.execute_script("arguments[0].click();", t)
                break
                
        time.sleep(2)
        
        # Click Run Guardrail
        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for b in buttons:
            if "Run Guardrail" in b.text:
                driver.execute_script("arguments[0].click();", b)
                break
                
        time.sleep(10)
        
        # Extract text from markdown elements
        elements = driver.find_elements(By.CSS_SELECTOR, ".stMarkdown, .stAlert")
        output = ""
        for e in elements:
            text = e.text.strip()
            if text and ("Code generation blocked" in text or "Code generation allowed" in text or "Stage 1 Classification:" in text or "ML evidence:" in text or "equations_or_formulas" in text or "algorithm_or_architecture" in text or "parameters_or" in text or "training_or" in text or "dataset_or" in text):
                output += text + "\n"
                print(f"[UI RENDERED]: {text}")
        
        with open("ui_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- {os.path.basename(pdf_path)} ---\n{output}")
            
        driver.quit()
    except Exception as e:
        print(f"Error testing {pdf_path}: {e}")

if os.path.exists("ui_output.txt"):
    os.remove("ui_output.txt")
    
p1 = r"D:\SENTRA AI\data\sessions\c77116a9-8510-444d-adb6-406d292ac5d2\uploads\1f9ff8d0-0213-45d8-b191-245b5a6e720e_attention.pdf"
p2 = r"D:\SENTRA AI\data\sessions\37d20f0d-1caf-4f3f-883e-09929e9d1ef8\uploads\78d15ba7-2b53-4809-8e15-a829238b0c16_Belief Maintenance in Bayesian Networks.pdf"
p3 = r"D:\SENTRA AI\data\sessions\489eb6fb-194c-4812-8578-8d23ecc654c5\uploads\886d664f-a3f5-4dd0-a196-373e188fd21a_DETECTION OF IRREGULAR, SUB-MM OPAQUE STRUCTURES IN THE ORION MOLECULAR CLOUDS.pdf"

test_ui(p1)
test_ui(p2)
test_ui(p3)
