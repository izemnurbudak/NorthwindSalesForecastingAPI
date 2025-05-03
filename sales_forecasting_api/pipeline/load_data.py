import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine("postgresql+psycopg2://postgres:yourpassword@localhost/yourdbname")

def load_data():
    try:
        # Load the data
        orders = pd.read_sql("SELECT * FROM orders", engine)
        order_details = pd.read_sql("SELECT * FROM order_details", engine)
        products = pd.read_sql("SELECT * FROM products", engine)
        categories = pd.read_sql("SELECT * FROM categories", engine)
        customers = pd.read_sql("SELECT * FROM customers", engine)
        
        query = """
        SELECT o.order_date, 
        od.product_id, 
        od.quantity, o.customer_id,
        p.product_name, 
        od.unit_price
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON p.product_id = od.product_id;
        """
        
        # Load the merged dataset
        df = pd.read_sql(query, engine)
        
        # Return the data
        return df, orders, order_details, products, categories, customers
    
    except Exception as e:
        raise ValueError(f"Data loading error: {str(e)}")
