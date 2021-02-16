#pragma once

#include "ParallelPhysics/ServerProtocol.h"
#include "ParallelPhysics/PPhHelpers.h"
#include <array>
#include <atomic>
#include <vector>
#include <memory>

constexpr uint32_t SECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND;  // quantum of time
constexpr uint32_t MILLISECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000;  // quantum of time
constexpr uint32_t IRRITATION_MULTIPLIER = 1'000; // purpose is convenient work with integral numbers
constexpr uint32_t FADING_VAL = IRRITATION_MULTIPLIER / MILLISECOND_IN_QUANTS;

class Synapse
{
public:
	Synapse(class Neuron *from);
	bool IsActive() const;
	uint32_t ReadAxon() const; // returns connected neuron activity value
private:
	class Neuron *m_from;
};

typedef std::vector<Synapse> SynapseVector;

class MotorSynapse
{
public:
	MotorSynapse(class MotorNeuron *to);
	uint32_t GetWeight() const;
	bool IsActive() const;
	void TransferIrritation(uint32_t irritation) const; // transfer irritation to motor neuron dendrite
private:
	class MotorNeuron *m_to;
	uint32_t m_weight = 1000;
};

typedef std::vector<MotorSynapse> MotorSynapseVector;

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

	virtual void Init() {}
	virtual void Tick() {}

	virtual bool IsActive() const;
	virtual uint32_t ReadAxon() const { return 0; }

	virtual uint8_t GetType() = 0;
	uint32_t GetId();

protected:
	enum class NeuronTypes
	{
		None = 0,
		SensoryNeuron,
		SensoryNeuronRed,
		SensoryNeuronGreen,
		SensoryNeuronBlue,
		MotorNeuron,
		SimpleAdderNeuron,
		EmptinessActivatorNeuron,
		PremotorNeuron,
		ReinforcementTransferNeuron
	};
};

class SensoryNeuron : public Neuron
{
public:
	SensoryNeuron() = default;
	virtual ~SensoryNeuron() = default;

	void Init() override;

	bool IsActive() const override;
	uint32_t ReadAxon() const override;
	void WriteDendrite(uint8_t exitation);
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;


private:
	uint8_t m_dendrite[2];
	uint8_t m_axon[2];
};

class SensoryNeuronRed : public SensoryNeuron
{
public:
	SensoryNeuronRed() = default;
	virtual ~SensoryNeuronRed() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronRed); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
};

class SensoryNeuronGreen : public SensoryNeuron
{
public:
	SensoryNeuronGreen() = default;
	virtual ~SensoryNeuronGreen() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronGreen); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
};

class SensoryNeuronBlue : public SensoryNeuron
{
public:
	SensoryNeuronBlue() = default;
	virtual ~SensoryNeuronBlue() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronBlue); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	bool IsActive() const override;
};

class MotorNeuron : public Neuron
{
public:
	MotorNeuron() = default;
	virtual ~MotorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init() override;

	bool IsActive() const override;

	void Tick() override;

	void AddIrritationToDendrite(uint16_t addedIrritation);
private:
	std::atomic<uint32_t> m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	bool m_isActive[2];
};

class SimpleAdderNeuron : public Neuron
{
public:
	SimpleAdderNeuron() = default;
	virtual ~SimpleAdderNeuron() = default;

	void InitExplicit(SynapseVector &synapses);

	uint32_t ReadAxon() const override;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SimpleAdderNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;


private:
	uint16_t m_axon[2];
	SynapseVector m_synapses;
};

class EmptinessActivatorNeuron : public Neuron
{
public:
	EmptinessActivatorNeuron() = default;
	virtual ~EmptinessActivatorNeuron() = default;

	void InitExplicit(SynapseVector &synapses);

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::EmptinessActivatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;

	bool IsActive() const override;
	uint32_t ReadAxon() const override;
private:
	SynapseVector m_synapses;
	uint32_t m_synapseIndex = 0;
	bool m_isActive[2];
	uint32_t m_activityTime = 0;
	uint32_t m_inhibitionTime = 0;
	const uint32_t m_activityTimeMax = 1000 * MILLISECOND_IN_QUANTS;
	const uint32_t m_inhibitionTimeMax = 1000 * MILLISECOND_IN_QUANTS;
};

class PremotorNeuron : public Neuron
{
public:
	PremotorNeuron() = default;
	virtual ~PremotorNeuron() = default;

	void InitExplicit(SynapseVector &&synapses, MotorSynapseVector &&motorSynapses);

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::PremotorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
private:
	SynapseVector m_synapses;
	MotorSynapseVector m_motorSynapses;
	uint16_t m_axon[2];
	uint32_t m_activatedSynapseIndex = 0;
};

typedef std::array<uint32_t, 8> TransferMotivationArray; // left, right, up, down and diagonals

class ReinforcementTransferNeuron : public Neuron
{
public:
	ReinforcementTransferNeuron() = default;
	virtual ~ReinforcementTransferNeuron() = default;

	void InitExplicit(PremotorNeuron *premotorNeuron, const PPh::VectorInt32Math &pos3D);

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ReinforcementTransferNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	
	PPh::VectorInt32Math GetPosition() const { return m_pos3D; }

private:
	void TransferMotivation(ReinforcementTransferNeuron* neighbour, uint32_t motivation);

	PremotorNeuron *m_premotorNeuron;
	PPh::VectorInt32Math m_pos3D;
	std::array <TransferMotivationArray, 2> m_transferMotivation;
	uint32_t m_internalMotivation = 0;
};
