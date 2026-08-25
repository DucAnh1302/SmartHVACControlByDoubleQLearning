# SmartHVACControlByDoubleQLearning
A Machine Learning project applying Double Q-Learning for intelligent HVAC control in smart buildings.

Một dự án Học máy (Machine Learning) ứng dụng thuật toán Double Q-Learning để điều khiển hệ thống HVAC (sưởi, thông gió và điều hòa không khí) một cách thông minh trong các tòa nhà thông minh.

## Project Overview

This project investigates the use of Reinforcement Learning, particularly Double Q-Learning, to develop an intelligent HVAC control agent for building environments.

Dự án này nghiên cứu việc sử dụng Học tăng cường (Reinforcement Learning), cụ thể là thuật toán Double Q-Learning, để phát triển một tác tử (agent) điều khiển HVAC thông minh cho môi trường tòa nhà.

The agent will learn to make HVAC control decisions by balancing thermal comfort and energy consumption.

Máy sẽ học cách đưa ra các quyết định điều khiển HVAC bằng cách cân bằng giữa sự thoải mái về nhiệt và mức tiêu thụ năng lượng.

The project uses the CU-BEMS (Smart Building Energy and IAQ) dataset, which contains building energy consumption and indoor environmental measurements collected across multiple floors and years.

Dự án sử dụng bộ dữ liệu CU-BEMS (Dữ liệu Năng lượng và Chất lượng không khí trong nhà - IAQ của Tòa nhà Thông minh), bao gồm thông tin về mức tiêu thụ năng lượng của tòa nhà và các chỉ số môi trường trong nhà được thu thập qua nhiều tầng và nhiều năm.

## Main Objective

The main objective is to design, implement, and evaluate a Double Q-Learning agent capable of learning HVAC control policies from building data.

Mục tiêu chính là thiết kế, triển khai và đánh giá một tác tử Double Q-Learning có khả năng học các chính sách điều khiển HVAC từ dữ liệu tòa nhà.

The project focuses on the following objectives:

- Analyze and understand the CU-BEMS dataset.
- Preprocess and transform the raw building data.
- Design the Reinforcement Learning environment.
- Define the State, Action, and Reward representations.
- Implement a Q-Learning baseline.
- Implement Double Q-Learning.
- Compare the two approaches.
- Evaluate thermal comfort and energy efficiency.
- Build a simple application to demonstrate the learned HVAC control policy.

Tiếng việt:

- Phân tích và tìm hiểu bộ dữ liệu CU-BEMS.
- Tiền xử lý và chuyển đổi dữ liệu tòa nhà thô.
- Thiết kế môi trường Học tăng cường.
- Xác định các biểu diễn Trạng thái (State), Hành động (Action) và Phần thưởng (Reward).
- Triển khai mô hình cơ sở (baseline) sử dụng Q-Learning.
- Triển khai thuật toán Double Q-Learning.
- So sánh hai phương pháp.
- Đánh giá mức độ thoải mái về nhiệt và hiệu quả sử dụng năng lượng.
- Xây dựng một ứng dụng đơn giản để minh họa chính sách điều khiển HVAC đã học được.

## Dataset

**Dataset:** CU-BEMS – Smart Building Energy and IAQ Data

The dataset contains building-level energy consumption and indoor environmental measurements collected across multiple floors and years.

Bộ dữ liệu bao gồm thông tin tiêu thụ năng lượng cấp tòa nhà và các chỉ số môi trường trong nhà được thu thập qua nhiều tầng và nhiều năm.

The project will use the available datasets while keeping the original raw data outside version control.

Dự án sẽ sử dụng các bộ dữ liệu có sẵn trong khi vẫn giữ dữ liệu thô gốc tách biệt khỏi hệ thống quản lý phiên bản.

Dataset source:

https://www.kaggle.com/datasets/claytonmiller/cubems-smart-building-energy-and-iaq-data

## Project Roadmap

### Phase 1 — Project Setup
Establish the project structure, documentation, development environment, and version control workflow.
Xây dựng cấu trúc dự án, tài liệu, môi trường phát triển và quy trình quản lý phiên bản.

### Phase 2 — Dataset Exploration
Analyze the available datasets, building structure, zones, HVAC systems, sensors, time-series characteristics, missing values, and energy consumption patterns.
Phân tích các bộ dữ liệu có sẵn, cấu trúc tòa nhà, các khu vực (zones), hệ thống HVAC, cảm biến, đặc điểm chuỗi thời gian, các giá trị bị thiếu và mô hình tiêu thụ năng lượng.

### Phase 3 — Data Preprocessing
Clean, transform, synchronize, normalize, and prepare the data for Reinforcement Learning.
Làm sạch, chuyển đổi, đồng bộ hóa, chuẩn hóa và chuẩn bị dữ liệu cho Học tăng cường.

### Phase 4 — State, Action, and Reward Design
Define the State representation, available HVAC Actions, and Reward function based on thermal comfort and energy consumption.
Xác định biểu diễn Trạng thái, các Hành động HVAC khả thi và hàm Phần thưởng dựa trên sự thoải mái về nhiệt và mức tiêu thụ năng lượng.

### Phase 5 — Environment Development
Build a data-driven simulation environment that represents the building HVAC control problem.
Xây dựng môi trường mô phỏng dựa trên dữ liệu để mô hình hóa bài toán điều khiển HVAC của tòa nhà.

### Phase 6 — Q-Learning Baseline
Implement a standard Q-Learning agent as a baseline for comparison.
Triển khai một module Q-Learning tiêu chuẩn làm cơ sở để so sánh.

### Phase 7 — Double Q-Learning
Implement and train the Double Q-Learning agent.
Triển khai và huấn luyện máy bằng Double Q-Learning.

### Phase 8 — Evaluation and Comparison
Evaluate learning performance, reward convergence, thermal comfort, energy consumption, and compare Q-Learning with Double Q-Learning.
Đánh giá hiệu suất học tập, sự hội tụ của phần thưởng, mức độ tiện nghi nhiệt, mức tiêu thụ năng lượng, đồng thời so sánh Q-Learning với Double Q-Learning.

### Phase 9 — Application
Develop a simple application to demonstrate the trained agent and its HVAC control decisions.
Phát triển một ứng dụng đơn giản để minh họa module đã được huấn luyện và các quyết định điều khiển hệ thống HVAC của nó.

### Phase 10 — Final Documentation
Prepare experiments, visualizations, results, conclusions, and final project documentation.
Chuẩn bị các thí nghiệm, hình ảnh trực quan hóa, kết quả, kết luận và tài liệu hoàn chỉnh của dự án.

## Project Structure

The repository is organized into separate folders for data, experimentation, source code, models, results, documentation, and application development.
Kho lưu trữ được tổ chức thành các thư mục riêng biệt cho dữ liệu, thử nghiệm, mã nguồn, mô hình, kết quả, tài liệu và phát triển ứng dụng.

Each folder contains its own README describing its responsibility and expected contents.

## Status

**Current Phase:** Project Setup

The project is currently establishing its structure and development plan before implementation begins.