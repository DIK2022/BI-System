# utils.py
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
import json
from PyQt6.QtWidgets import QApplication


class DataProcessor:
    """Обработчик данных для Pandas и Polars"""
    
    @staticmethod
    def analyze_pandas(df):
        """Анализ данных в Pandas"""
        analysis = []
        analysis.append("=" * 50)
        analysis.append("АНАЛИЗ ДАННЫХ (Pandas)")
        analysis.append("=" * 50 + "\n")
        
        # Основная информация
        analysis.append("1. ОСНОВНАЯ ИНФОРМАЦИЯ:")
        analysis.append(f"   Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
        analysis.append(f"   Типы данных:\n{df.dtypes}")
        
        # Пропущенные значения
        analysis.append("\n2. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        for col in df.columns:
            analysis.append(f"   {col}: {missing[col]} ({missing_pct[col]}%)")
            
        # Основные статистики
        analysis.append("\n3. СТАТИСТИКА ДЛЯ ЧИСЛОВЫХ КОЛОНОК:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats = df[col].describe()
            analysis.append(f"   {col}:")
            analysis.append(f"     Мин: {stats['min']:.2f}")
            analysis.append(f"     Макс: {stats['max']:.2f}")
            analysis.append(f"     Среднее: {stats['mean']:.2f}")
            analysis.append(f"     Медиана: {stats['50%']:.2f}")
            analysis.append(f"     Std: {stats['std']:.2f}")
            
        # Категориальные колонки
        analysis.append("\n4. КАТЕГОРИАЛЬНЫЕ КОЛОНКИ:")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            unique_vals = df[col].nunique()
            top_vals = df[col].value_counts().head(3)
            analysis.append(f"   {col}:")
            analysis.append(f"     Уникальных значений: {unique_vals}")
            analysis.append(f"     Топ-3: {dict(top_vals)}")
            
        return "\n".join(analysis)
    
    @staticmethod
    def analyze_polars(df):
        """Анализ данных в Polars"""
        analysis = []
        analysis.append("=" * 50)
        analysis.append("АНАЛИЗ ДАННЫХ (Polars)")
        analysis.append("=" * 50 + "\n")
        
        # Основная информация
        analysis.append("1. ОСНОВНАЯ ИНФОРМАЦИЯ:")
        analysis.append(f"   Размер: {df.height} строк, {df.width} колонок")
        
        # Схема данных
        analysis.append("2. СХЕМА ДАННЫХ:")
        for col, dtype in zip(df.columns, df.dtypes):
            analysis.append(f"   {col}: {dtype}")
            
        # Пропущенные значения
        analysis.append("\n3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = (null_count / df.height * 100)
            analysis.append(f"   {col}: {null_count} ({null_pct:.2f}%)")
            
        # Основные статистики
        analysis.append("\n4. СТАТИСТИКА ДЛЯ ЧИСЛОВЫХ КОЛОНОК:")
        numeric_cols = [col for col in df.columns 
                       if str(df[col].dtype) in ['Int64', 'Int32', 'Int16', 'Int8',
                                                'UInt64', 'UInt32', 'UInt16', 'UInt8',
                                                'Float64', 'Float32']]
        
        for col in numeric_cols:
            stats = df[col].describe()
            analysis.append(f"   {col}:")
            analysis.append(f"     Мин: {stats['min'][0]:.2f}")
            analysis.append(f"     Макс: {stats['max'][0]:.2f}")
            analysis.append(f"     Среднее: {stats['mean'][0]:.2f}")
            analysis.append(f"     Медиана: {stats['median'][0]:.2f}")
            analysis.append(f"     Std: {stats['std'][0]:.2f}")
            
        # Категориальные колонки
        analysis.append("\n5. КАТЕГОРИАЛЬНЫЕ КОЛОНКИ:")
        cat_cols = [col for col in df.columns 
                   if str(df[col].dtype) in ['Utf8', 'Categorical']]
        
        for col in cat_cols:
            unique_vals = df[col].n_unique()
            value_counts = df[col].value_counts().sort('count', descending=True)
            top_vals = {row[col]: row['count'] 
                       for row in value_counts.head(3).to_dicts()}
            
            analysis.append(f"   {col}:")
            analysis.append(f"     Уникальных значений: {unique_vals}")
            analysis.append(f"     Топ-3: {top_vals}")
            
        return "\n".join(analysis)
    
    @staticmethod
    def get_correlation_pandas(df):
        """Матрица корреляций для Pandas"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return "Недостаточно числовых колонок для корреляции"
        return numeric_df.corr()
    
    @staticmethod
    def get_correlation_polars(df):
        """Матрица корреляций для Polars"""
        numeric_cols = [col for col in df.columns 
                       if str(df[col].dtype) in ['Float64', 'Float32', 
                                                'Int64', 'Int32', 'Int16', 'Int8']]
        
        if len(numeric_cols) < 2:
            return "Недостаточно числовых колонок для корреляции"
            
        # Вычисляем корреляции попарно
        correlations = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = df[[col1, col2]].corr()
                if len(corr) > 0:
                    correlations.append(f"{col1} - {col2}: {corr.row(0)[1]:.3f}")
                    
        return "\n".join(correlations) if correlations else "Нет данных"
    
    @staticmethod
    def describe_pandas(df):
        """Статистическое описание для Pandas"""
        return df.describe(include='all')
    
    @staticmethod
    def describe_polars(df):
        """Статистическое описание для Polars"""
        return df.describe()
    
    @staticmethod
    def groupby_pandas(df, group_col, agg_col, agg_func):
        """Группировка для Pandas"""
        if agg_func == 'count':
            return df.groupby(group_col)[agg_col].count()
        else:
            return df.groupby(group_col)[agg_col].agg(agg_func)
    
    @staticmethod
    def groupby_polars(df, group_col, agg_col, agg_func):
        """Группировка для Polars"""
        if agg_func == 'count':
            return df.group_by(group_col).agg(pl.count(agg_col))
        else:
            agg_expr = getattr(pl, agg_func)(agg_col)
            return df.group_by(group_col).agg(agg_expr)


class DataConverter:
    """Конвертер между Pandas и Polars"""
    
    @staticmethod
    def pandas_to_polars(df_pd):
        """Конвертирует Pandas DataFrame в Polars"""
        return pl.from_pandas(df_pd)
    
    @staticmethod
    def polars_to_pandas(df_pl):
        """Конвертирует Polars DataFrame в Pandas"""
        return df_pl.to_pandas()


class StyleManager:
    """Менеджер стилей приложения"""
    
    @staticmethod
    def apply_style(widget, theme='light'):
        """Применяет тему к приложению"""
        if theme == 'dark':
            style = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTableView {
                background-color: #3c3c3c;
                alternate-background-color: #4a4a4a;
                gridline-color: #555555;
                selection-background-color: #0078d7;
            }
            QHeaderView::section {
                background-color: #404040;
                padding: 5px;
                border: 1px solid #555555;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #404040;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d7;
            }
            QToolBar {
                background-color: #404040;
                border: none;
                spacing: 5px;
            }
            QToolButton {
                padding: 5px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #505050;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
                min-height: 25px;
            }
            QMenuBar {
                background-color: #404040;
            }
            QMenuBar::item:selected {
                background-color: #505050;
            }
            QMenu {
                background-color: #404040;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #0078d7;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
            }
            """
            
        elif theme == 'blue':
            style = """
            QMainWindow {
                background-color: #f0f8ff;
            }
            QWidget {
                background-color: #f0f8ff;
                color: #003366;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTableView {
                background-color: white;
                alternate-background-color: #f0f8ff;
                gridline-color: #cce0ff;
                selection-background-color: #3399ff;
            }
            QHeaderView::section {
                background-color: #66b3ff;
                color: white;
                padding: 5px;
                border: 1px solid #3399ff;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 2px solid #66b3ff;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #cce0ff;
                color: #003366;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #66b3ff;
                color: white;
                font-weight: bold;
            }
            QToolBar {
                background-color: #66b3ff;
                border: none;
                spacing: 5px;
            }
            QToolButton {
                padding: 5px;
                border-radius: 3px;
                background-color: #99ccff;
            }
            QToolButton:hover {
                background-color: #3399ff;
            }
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004d99;
            }
            QPushButton:pressed {
                background-color: #003366;
            }
            """
            
        else:  # light theme
            style = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTableView {
                background-color: white;
                alternate-background-color: #f9f9f9;
                gridline-color: #e0e0e0;
                selection-background-color: #2196F3;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #e0e0e0;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QToolBar {
                background-color: #f8f8f8;
                border: none;
                spacing: 5px;
            }
            QToolButton {
                padding: 5px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                padding: 5px;
                border-radius: 3px;
                min-height: 25px;
            }
            QMenuBar {
                background-color: #f8f8f8;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }
            QMenu {
                background-color: white;
                border: 1px solid #e0e0e0;
            }
            QMenu::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
            """
            
        widget.setStyleSheet(style)