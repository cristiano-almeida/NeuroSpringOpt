import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from mealpy import FloatVar, Problem, ES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class NeuralNetwork:
    """Implementação completa de rede neural feedforward para regressão"""
    
    def __init__(self, layer_sizes, gradient_modifier_function):
        """
        Inicializa a rede neural com arquitetura especificada
        
        Args:
            layer_sizes (list): Lista com número de neurônios em cada camada
            gradient_modifier_function (function): Função para modificar gradientes durante o backpropagation
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.gradient_modifier = gradient_modifier_function
        
        # Inicialização dos pesos com Xavier/Glorot initialization
        self.biases = [np.random.randn(neurons, 1) for neurons in layer_sizes[1:]]
        self.weights = [np.random.randn(neurons, input_size)/np.sqrt(input_size) 
                       for input_size, neurons in zip(layer_sizes[:-1], layer_sizes[1:])]
        
        # Histórico de treinamento
        self.training_history = {
            'epoch': [],
            'training_loss': [],
            'validation_loss': [],
            'training_accuracy': [],
            'validation_accuracy': []
        }

    def train(self, training_data, validation_data, num_epochs, batch_size, learning_rate, regularization_lambda):
        """
        Treina a rede neural usando SGD com momentum
        
        Args:
            training_data (list): Lista de tuplas (input, target)
            validation_data (list): Dados para validação
            num_epochs (int): Número de épocas de treinamento
            batch_size (int): Tamanho do mini-batch
            learning_rate (float): Taxa de aprendizagem
            regularization_lambda (float): Parâmetro de regularização L2
        """
        num_training_samples = len(training_data)
        
        for epoch in range(num_epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[batch_start:batch_start + batch_size]
                for batch_start in range(0, num_training_samples, batch_size)
            ]
            
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, learning_rate, regularization_lambda, num_training_samples)
            
            # Avaliação após cada época
            training_loss, training_accuracy = self.evaluate(training_data)
            validation_loss, validation_accuracy = self.evaluate(validation_data)
            
            # Registro no histórico
            self.training_history['epoch'].append(epoch)
            self.training_history['training_loss'].append(training_loss)
            self.training_history['validation_loss'].append(validation_loss)
            self.training_history['training_accuracy'].append(training_accuracy)
            self.training_history['validation_accuracy'].append(validation_accuracy)
            
            # Log periódico
            if epoch % 50 == 0:
                print(f"Época {epoch}: Loss Treino={training_loss:.6f}, Loss Validação={validation_loss:.6f}")

        return self.training_history

    def update_parameters(self, mini_batch, learning_rate, regularization_lambda, num_training_samples):
        """Atualiza os parâmetros da rede usando backpropagation em um mini-batch"""
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        
        for input_data, target in mini_batch:
            delta_gradient_biases, delta_gradient_weights = self.backpropagation(input_data, target)
            gradient_biases = [gb + dgb for gb, dgb in zip(gradient_biases, delta_gradient_biases)]
            gradient_weights = [gw + dgw for gw, dgw in zip(gradient_weights, delta_gradient_weights)]
        
        # Atualização com regularização L2
        self.weights = [
            (1 - learning_rate * (regularization_lambda / num_training_samples)) * weight - 
            (learning_rate / len(mini_batch)) * gradient_weight
            for weight, gradient_weight in zip(self.weights, gradient_weights)
        ]
        
        self.biases = [
            bias - (learning_rate / len(mini_batch)) * gradient_bias
            for bias, gradient_bias in zip(self.biases, gradient_biases)
        ]

    def backpropagation(self, input_data, target):
        """Implementação completa do algoritmo backpropagation"""
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        
        # Feedforward
        activation = input_data
        activations = [input_data]  # Lista para armazenar todas as ativações
        weighted_inputs = []  # Lista para armazenar todos os z-vectors
        
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = np.dot(weight, activation) + bias
            weighted_inputs.append(weighted_input)
            activation = sigmoid(weighted_input)
            activations.append(activation)
        
        # Backward pass
        error = activations[-1] - target
        delta = error * self.gradient_modifier(weighted_inputs[-1])
        
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.dot(delta, activations[-2].transpose())
        
        for layer_index in range(2, self.num_layers):
            weighted_input = weighted_inputs[-layer_index]
            activation_derivative = sigmoid_derivative(weighted_input)
            delta = np.dot(self.weights[-layer_index + 1].transpose(), delta) * activation_derivative
            delta = delta * self.gradient_modifier(weighted_input)
            
            gradient_biases[-layer_index] = delta
            gradient_weights[-layer_index] = np.dot(delta, activations[-layer_index - 1].transpose())
        
        return (gradient_biases, gradient_weights)

    def predict(self, input_data):
        """Faz uma predição para uma única entrada"""
        activation = input_data
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = np.dot(weight, activation) + bias
            activation = sigmoid(weighted_input)
        return activation[0, 0]  # Retorna valor escalar para regressão

    def evaluate(self, dataset):
        """Avalia o desempenho da rede em um conjunto de dados"""
        squared_errors = []
        correct_predictions = 0
        
        for input_data, target in dataset:
            prediction = self.predict(input_data)
            
            # Calcula erro quadrático médio
            target_value = target[0, 0]
            squared_errors.append((prediction - target_value)**2)
            
            # Considera acerto se erro absoluto < 10% do valor alvo
            if abs(prediction - target_value) < 0.1 * abs(target_value):
                correct_predictions += 1
        
        mean_squared_error = np.mean(squared_errors)
        accuracy = correct_predictions / len(dataset)
        
        return mean_squared_error, accuracy


class SpringOptimizer:
    """Sistema completo de otimização de mola helicoidal usando redes neurais"""
    
    def __init__(self, output_directory='spring_optimization_results'):
        # Configurações do problema da mola
        self.design_variables_bounds = [
            FloatVar(lb=0.05, ub=2.0, name="wire_diameter"),   # d
            FloatVar(lb=0.25, ub=1.3, name="coil_diameter"),    # D
            FloatVar(lb=2.0, ub=15.0, name="number_of_coils")   # N
        ]
        
        # Solução de referência do artigo
        self.reference_solution = {
            'wire_diameter': 0.051796393,
            'coil_diameter': 0.359305355,
            'number_of_coils': 11.138859,
            'spring_weight': 0.012665443
        }
        
        # Configurações da rede neural
        self.network_architecture = [3, 32, 16, 1]  # Arquitetura: 3 entradas, 2 camadas ocultas (32 e 16), 1 saída
        self.training_parameters = {
            'number_of_epochs': 500,
            'batch_size': 64,
            'learning_rate': 0.005,
            'regularization_lambda': 0.01
        }
        
        # Configurações do algoritmo de otimização
        self.optimization_parameters = {
            'maximum_epochs': 2000,
            'population_size': 300,
            'offspring_ratio': 0.7
        }
        
        # Funções modificadoras de gradiente
        self.gradient_modifiers = {
            "Original": lambda weighted_input: 1,
            "Tanh": lambda weighted_input: 1 + 0.15 * np.tanh(0.05 * weighted_input**2),
            "ArtigoADAM": lambda weighted_input: 1 + 0.2 * (1 - np.exp(-0.1 * weighted_input**2)),
            "Proposto": lambda weighted_input: 1 + 0.1 * (1 - np.tanh(0.03 * weighted_input**2)) + 0.05 * np.sin(weighted_input)
        }
        
        # Configuração do sistema de arquivos
        self.setup_file_system(output_directory)
        
        # Componentes do sistema
        self.data_scaler = StandardScaler()
        self.neural_network = None
        self.optimization_algorithm = None
        self.target_statistics = {'minimum': None, 'maximum': None}

    def setup_file_system(self, base_directory):
        """Configura a estrutura de diretórios para armazenar resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_directory = os.path.join(base_directory, f"optimization_{timestamp}")
        
        # Cria diretórios necessários
        self.plots_directory = os.path.join(self.base_output_directory, 'visualizations')
        self.models_directory = os.path.join(self.base_output_directory, 'trained_models')
        self.data_directory = os.path.join(self.base_output_directory, 'processed_data')
        
        for directory in [self.base_output_directory, self.plots_directory, 
                         self.models_directory, self.data_directory]:
            os.makedirs(directory, exist_ok=True)

    def generate_training_dataset(self, number_of_samples=20000):
        """Gera dataset sintético para treinamento da rede neural"""
        # Gera amostras com distribuição mista (normal + uniforme)
        wire_diameter_samples = np.concatenate([
            np.random.normal(0.052, 0.003, int(number_of_samples * 0.7)),
            np.random.uniform(0.05, 2.0, int(number_of_samples * 0.3))
        ])
        
        coil_diameter_samples = np.concatenate([
            np.random.normal(0.36, 0.02, int(number_of_samples * 0.7)),
            np.random.uniform(0.25, 1.3, int(number_of_samples * 0.3))
        ])
        
        number_of_coils_samples = np.concatenate([
            np.random.normal(11.0, 1.5, int(number_of_samples * 0.7)),
            np.random.uniform(2.0, 15.0, int(number_of_samples * 0.3))
        ])
        
        # Combina as amostras
        design_variables = np.column_stack((
            wire_diameter_samples, 
            coil_diameter_samples, 
            number_of_coils_samples
        ))
        
        # Calcula o peso correspondente
        spring_weights = np.array([self.calculate_spring_weight(x) for x in design_variables])
        
        # Filtra para manter apenas soluções viáveis
        feasibility_mask = (spring_weights < 0.02) & np.array([
            all(constraint <= 0 for constraint in self.calculate_design_constraints(x))
            for x in design_variables
        ])
        
        filtered_design_variables = design_variables[feasibility_mask]
        filtered_spring_weights = spring_weights[feasibility_mask]
        
        # Salva os dados gerados
        dataset_dataframe = pd.DataFrame(
            filtered_design_variables, 
            columns=['wire_diameter', 'coil_diameter', 'number_of_coils']
        )
        dataset_dataframe['spring_weight'] = filtered_spring_weights
        dataset_dataframe.to_csv(
            os.path.join(self.data_directory, 'training_dataset.csv'), 
            index=False
        )
        
        return filtered_design_variables, filtered_spring_weights

    def calculate_spring_weight(self, design_variables):
        """Calcula o peso da mola para um dado conjunto de parâmetros"""
        wire_diameter, coil_diameter, number_of_coils = design_variables
        return (number_of_coils + 2) * coil_diameter * wire_diameter**2

    def calculate_design_constraints(self, design_variables):
        """Calcula as restrições de projeto para a mola helicoidal"""
        wire_diameter, coil_diameter, number_of_coils = design_variables
        epsilon = 1e-10  # Para evitar divisão por zero
        
        return [
            # Restrição de tensão de cisalhamento
            1 - (coil_diameter**3 * number_of_coils) / (71785 * wire_diameter**4 + epsilon),
            
            # Restrição de frequência natural
            (4 * coil_diameter**2 - wire_diameter * coil_diameter) / 
            (12566 * (coil_diameter * wire_diameter**3 - wire_diameter**4 + epsilon)) + 
            1 / (5108 * wire_diameter**2) - 1,
            
            # Restrição de comprimento
            1 - (140.45 * wire_diameter) / (coil_diameter**2 * number_of_coils + epsilon),
            
            # Restrição geométrica
            (wire_diameter + coil_diameter) / 1.5 - 1
        ]

    def train_and_evaluate_network(self, input_data, target_data, gradient_modifier_name):
        """Treina e avalia a rede neural com um modificador de gradiente específico"""
        # Normalização dos dados
        normalized_input = self.data_scaler.fit_transform(input_data)
        normalized_target = (target_data - target_data.min()) / (target_data.max() - target_data.min())
        self.target_statistics = {
            'minimum': target_data.min(),
            'maximum': target_data.max()
        }
        
        # Divisão treino/validação
        (train_input, validation_input, 
         train_target, validation_target) = train_test_split(
            normalized_input, normalized_target, 
            test_size=0.2, 
            random_state=42
        )
        
        # Prepara os dados no formato correto
        training_dataset = [
            (input.reshape(-1, 1), np.array([target]).reshape(-1, 1))
            for input, target in zip(train_input, train_target)
        ]
        
        validation_dataset = [
            (input.reshape(-1, 1), np.array([target]).reshape(-1, 1))
            for input, target in zip(validation_input, validation_target)
        ]
        
        # Cria e treina a rede neural
        self.neural_network = NeuralNetwork(
            layer_sizes=self.network_architecture,
            gradient_modifier_function=self.gradient_modifiers[gradient_modifier_name]
        )
        
        training_history = self.neural_network.train(
            training_data=training_dataset,
            validation_data=validation_dataset,
            num_epochs=self.training_parameters['number_of_epochs'],
            batch_size=self.training_parameters['batch_size'],
            learning_rate=self.training_parameters['learning_rate'],
            regularization_lambda=self.training_parameters['regularization_lambda']
        )
        
        # Salva resultados do treinamento
        self.save_training_artifacts(training_history, gradient_modifier_name)
        
        return training_history

    def save_training_artifacts(self, history, modifier_name):
        """Salva todos os artefatos do treinamento (modelo, gráficos, dados)"""
        # Salva histórico em CSV
        history_dataframe = pd.DataFrame(history)
        history_dataframe.to_csv(
            os.path.join(self.data_directory, f'training_history_{modifier_name}.csv'), 
            index=False
        )
        
        # Gráficos de desempenho
        self.plot_training_metrics(history, modifier_name)
        
        # Salva o modelo treinado
        model_parameters = {
            'weights': self.neural_network.weights,
            'biases': self.neural_network.biases,
            'scaler_mean': self.data_scaler.mean_,
            'scaler_scale': self.data_scaler.scale_,
            'target_min': self.target_statistics['minimum'],
            'target_max': self.target_statistics['maximum']
        }
        
        np.save(
            os.path.join(self.models_directory, f'model_parameters_{modifier_name}.npy'), 
            model_parameters
        )

    def plot_training_metrics(self, history, modifier_name):
        """Gera e salva gráficos das métricas de treinamento"""
        plt.figure(figsize=(14, 6))
        
        # Gráfico de loss
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['training_loss'], label='Treino')
        plt.plot(history['epoch'], history['validation_loss'], label='Validação')
        plt.title(f'Função de Perda - {modifier_name}', pad=20)
        plt.xlabel('Época', labelpad=10)
        plt.ylabel('Loss (MSE)', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['training_accuracy'], label='Treino')
        plt.plot(history['epoch'], history['validation_accuracy'], label='Validação')
        plt.title(f'Acurácia - {modifier_name}', pad=20)
        plt.xlabel('Época', labelpad=10)
        plt.ylabel('Acurácia', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_directory, f'training_metrics_{modifier_name}.png'))
        plt.close()

    def optimize_design_with_network(self, gradient_modifier_name):
        """Executa a otimização do projeto usando a rede neural treinada"""
        optimization_problem = Problem(
            bounds=self.design_variables_bounds,
            minmax="min",
            obj_func=self.network_objective_function
        )
        
        self.optimization_algorithm = ES.OriginalES(
            epoch=self.optimization_parameters['maximum_epochs'],
            pop_size=self.optimization_parameters['population_size'],
            lamda=self.optimization_parameters['offspring_ratio']
        )
        
        self.optimization_algorithm.solve(optimization_problem)
        
        return self.process_optimization_results(gradient_modifier_name)

    def network_objective_function(self, design_variables):
        """Função objetivo para a otimização incorporando restrições"""
        predicted_weight = self.predict_with_network(design_variables)
        constraints = self.calculate_design_constraints(design_variables)
        
        # Penalização quadrática para restrições violadas
        penalty = sum(max(0, constraint)**2 for constraint in constraints) * 1e6
        
        return predicted_weight + penalty

    def predict_with_network(self, design_variables):
        """Faz predição do peso usando a rede neural"""
        if None in [self.target_statistics['minimum'], self.target_statistics['maximum']]:
            raise ValueError("Estatísticas do target não disponíveis. Treine a rede primeiro.")
            
        normalized_input = self.data_scaler.transform(design_variables.reshape(1, -1))
        normalized_output = self.neural_network.predict(normalized_input.T)
        
        # Desnormaliza a saída
        return (self.target_statistics['minimum'] + 
                normalized_output * (self.target_statistics['maximum'] - self.target_statistics['minimum']))

    def process_optimization_results(self, modifier_name):
        """Processa e salva os resultados da otimização"""
        optimal_solution = self.optimization_algorithm.g_best.solution
        
        results = {
            'optimization_method': modifier_name,
            'optimal_wire_diameter': optimal_solution[0],
            'optimal_coil_diameter': optimal_solution[1],
            'optimal_number_of_coils': optimal_solution[2],
            'neural_network_prediction': self.predict_with_network(optimal_solution),
            'actual_spring_weight': self.calculate_spring_weight(optimal_solution),
            'reference_spring_weight': self.reference_solution['spring_weight'],
            'design_constraints': self.calculate_design_constraints(optimal_solution),
            'is_feasible': all(c <= 0 for c in self.calculate_design_constraints(optimal_solution)),
            'convergence_history': self.optimization_algorithm.history.list_global_best_fit
        }
        
        # Salva resultados em CSV
        results_dataframe = pd.DataFrame([results])
        results_dataframe.to_csv(
            os.path.join(self.data_directory, f'optimization_results_{modifier_name}.csv'), 
            index=False
        )
        
        # Gera gráficos de convergência
        self.plot_convergence_history(results['convergence_history'], modifier_name)
        
        return results

    def plot_convergence_history(self, convergence_data, modifier_name):
        """Gera gráfico da história de convergência da otimização"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(convergence_data, 'b-', linewidth=2, label='Otimização')
        plt.axhline(
            y=self.reference_solution['spring_weight'], 
            color='r', 
            linestyle='--', 
            label=f"Referência ({self.reference_solution['spring_weight']:.6f} kg)"
        )
        
        plt.title(f"Convergência da Otimização - {modifier_name}", pad=20)
        plt.xlabel("Iteração", labelpad=10)
        plt.ylabel("Peso da Mola (kg)", labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.plots_directory, f'convergence_{modifier_name}.png'))
        plt.close()

    def compare_gradient_modifiers(self):
        """Compara o desempenho de todos os modificadores de gradiente"""
        print("="*70)
        print("GERAÇÃO DO DATASET DE TREINAMENTO")
        print("="*70)
        design_variables, spring_weights = self.generate_training_dataset()
        
        comparison_results = []
        
        for modifier_name in self.gradient_modifiers.keys():
            print("\n" + "="*70)
            print(f"PROCESSANDO MÉTODO: {modifier_name}")
            print("="*70)
            
            # Fase de treinamento
            print("\nTreinamento da Rede Neural:")
            self.train_and_evaluate_network(design_variables, spring_weights, modifier_name)
            
            # Fase de otimização
            print("\nOtimização do Projeto:")
            results = self.optimize_design_with_network(modifier_name)
            comparison_results.append(results)
            
            # Exibe resultados intermediários
            self.display_optimization_results(results)
        
        # Gera análise comparativa final
        self.generate_comparative_analysis(comparison_results)
        
        return comparison_results

    def display_optimization_results(self, results):
        """Exibe os resultados formatados da otimização"""
        weight_difference = ((results['actual_spring_weight'] - results['reference_spring_weight']) / 
                           results['reference_spring_weight'] * 100)
        
        print("\n⭐ Resultados da Otimização:")
        print(f"Método: {results['optimization_method']}")
        print(f"Diâmetro do fio (d): {results['optimal_wire_diameter']:.8f} m")
        print(f"Diâmetro médio (D): {results['optimal_coil_diameter']:.8f} m")
        print(f"Número de espiras (N): {results['optimal_number_of_coils']:.8f}")
        print(f"Peso predito: {results['neural_network_prediction']:.8f} kg")
        print(f"Peso real: {results['actual_spring_weight']:.8f} kg")
        print(f"Peso de referência: {results['reference_spring_weight']:.8f} kg")
        print(f"Diferença: {weight_difference:.2f}%")
        print(f"Viável: {'Sim' if results['is_feasible'] else 'Não'}")
        print("Restrições:", [f"{c:.2e}" for c in results['design_constraints']])

    def generate_comparative_analysis(self, results_list):
        """Gera análise comparativa completa entre os métodos"""
        comparison_data = []
        
        for result in results_list:
            weight_difference = ((result['actual_spring_weight'] - result['reference_spring_weight']) / 
                               result['reference_spring_weight'] * 100)
            
            comparison_data.append({
                'Método': result['optimization_method'],
                'd (m)': result['optimal_wire_diameter'],
                'D (m)': result['optimal_coil_diameter'],
                'N': result['optimal_number_of_coils'],
                'Peso Real (kg)': result['actual_spring_weight'],
                'Diferença (%)': weight_difference,
                'Viável': result['is_feasible'],
                'Restrição 1': result['design_constraints'][0],
                'Restrição 2': result['design_constraints'][1],
                'Restrição 3': result['design_constraints'][2],
                'Restrição 4': result['design_constraints'][3]
            })
        
        # Salva tabela comparativa
        comparison_dataframe = pd.DataFrame(comparison_data)
        comparison_dataframe.to_csv(
            os.path.join(self.data_directory, 'final_comparison.csv'), 
            index=False
        )
        
        # Gera gráficos comparativos
        self.plot_comparative_results(comparison_dataframe)
        
        # Exibe resumo final
        self.display_final_comparison(comparison_dataframe)

    def plot_comparative_results(self, comparison_data):
        """Gera gráficos comparativos entre os métodos"""
        # Gráfico de comparação de pesos
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            comparison_data['Método'], 
            comparison_data['Peso Real (kg)'],
            color=['blue', 'green', 'orange', 'red']
        )
        
        plt.axhline(
            y=self.reference_solution['spring_weight'], 
            color='black', 
            linestyle='--', 
            linewidth=2,
            label=f"Referência ({self.reference_solution['spring_weight']:.6f} kg)"
        )
        
        # Adiciona valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.6f} kg',
                ha='center', 
                va='bottom'
            )
        
        plt.title("Comparação dos Pesos Obtidos por Método", pad=20)
        plt.ylabel("Peso da Mola (kg)", labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_directory, 'weight_comparison.png'))
        plt.close()
        
        # Gráfico de comparação de parâmetros
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        parameters = ['d (m)', 'D (m)', 'N']
        titles = [
            'Comparação do Diâmetro do Fio (d)',
            'Comparação do Diâmetro Médio (D)',
            'Comparação do Número de Espiras (N)'
        ]
        
        for ax, param, title in zip(axes, parameters, titles):
            ax.bar(
                comparison_data['Método'], 
                comparison_data[param],
                color=['blue', 'green', 'orange', 'red']
            )
            ax.set_title(title, pad=15)
            ax.set_ylabel(param)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_directory, 'parameters_comparison.png'))
        plt.close()

    def display_final_comparison(self, comparison_data):
        """Exibe a comparação final formatada"""
        print("\n" + "="*70)
        print("⭐ COMPARAÇÃO FINAL DOS MÉTODOS")
        print("="*70)
        
        for _, row in comparison_data.iterrows():
            print("\n" + "-"*60)
            print(f"Método: {row['Método']}")
            print(f"  d = {row['d (m)']:.8f} m | D = {row['D (m)']:.8f} m | N = {row['N']:.8f}")
            print(f"  Peso: {row['Peso Real (kg)']:.8f} kg | Diferença: {row['Diferença (%)']:.2f}%")
            print(f"  Viável: {'Sim' if row['Viável'] else 'Não'}")
            print("  Restrições:", [f"{row[f'Restrição {i+1}']:.2e}" for i in range(4)])


def sigmoid(z):
    """Função de ativação sigmoide"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivada da função sigmoide"""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    print("="*70)
    print("SISTEMA DE OTIMIZAÇÃO DE MOLA HELICOIDAL COM REDES NEURAIS")
    print("="*70)
    
    try:
        optimizer = SpringOptimizer(output_directory='spring_optimization_results')
        final_comparison = optimizer.compare_gradient_modifiers()
        
    except Exception as error:
        print("\n❌ Erro durante a execução:")
        print(f"Tipo: {type(error).__name__}")
        print(f"Detalhes: {str(error)}")
        print("\nVerifique os parâmetros e os dados de entrada.")