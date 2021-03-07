#pragma once

#include "ParallelPhysics/ServerProtocol.h"
#include "ParallelPhysics/PPhHelpers.h"
#include <array>
#include <atomic>
#include <vector>
#include <memory>

constexpr uint32_t SECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND;  // quantum of time
constexpr uint32_t MILLISECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1'000;  // quantum of time
constexpr uint32_t EXCITATION_MULTIPLIER = 1'000; // purpose: convenient work with integral numbers
constexpr uint32_t FADING_VAL = EXCITATION_MULTIPLIER / MILLISECOND_IN_QUANTS;

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

class InhibitorSynapse
{
public:
	InhibitorSynapse(class Neuron *to, uint32_t percentOfInhibition);
	void Inhibit(uint32_t val);
private:
	class Neuron *m_to;
	uint32_t m_percentOfInhibition;
};

typedef std::vector<InhibitorSynapse> InhibitorSynapseVector;

class MotorSynapse
{
public:
	MotorSynapse(class MotorNeuron *to);

	bool IsActive() const;
	uint32_t GetWeight() const;
	void AddWeight(uint32_t weight);
	void TransferExcitation(uint32_t excitation) const; // transfer irritation to motor neuron dendrite
	void HalfWeight();
private:
	class MotorNeuron *m_to;
	uint32_t m_weight = EXCITATION_MULTIPLIER;
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

	virtual void Inhibit(uint32_t) {};

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
		MotivationTransferNeuron,
		ActivatorNeuron,
		MotivationSourceNeuron,
		HungerActivatorNeuron
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
	void WriteDendrite(uint8_t excitation);
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
	void InitExplicit(InhibitorSynapseVector &&inhibitorSynapses);

	bool IsActive() const override;

	void Tick() override;

	void Inhibit(uint32_t addedInhibition) override;
	void AddExcitationToDendrite(uint16_t addedExcitation);
private:
	std::atomic<uint32_t> m_dendrite[2]; 
	std::atomic<uint32_t> m_inhibitor[2];
	bool m_isActive[2];
	InhibitorSynapseVector m_inhibitorSynapses;
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
	uint32_t m_axon[2];
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
	uint32_t ReadAxon() const override;
	void AddReinforcement(uint32_t reinforcement);
private:
	SynapseVector m_synapses;
	MotorSynapseVector m_motorSynapses;
	uint32_t m_axon[2];
	uint32_t m_activatedSynapseIndex = 0;
	uint32_t m_excitationMax = 0;
	uint64_t m_excitationTimeStart = 0;
};

typedef std::array<uint32_t, 8> TransferMotivationArray; // left, right, up, down and diagonals

class MotivationTransferNeuron : public Neuron
{
public:
	MotivationTransferNeuron() = default;
	virtual ~MotivationTransferNeuron() = default;

	void InitExplicit(const Neuron *reinforcementActivator, PremotorNeuron *premotorNeuron, const PPh::VectorInt32Math &pos3D);

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotivationTransferNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	
	uint32_t ReadAxon() const override;
	PPh::VectorInt32Math GetPosition() const { return m_pos3D; }
	PremotorNeuron* GetPremotorNeuron() const { return m_premotorNeuron; }

	void SetCentralMotivationSource(uint32_t m_motivation); // this neuron is the source of motivation. Will transfer it in all directions, but not on itself.
	void TransferCentralMotivation(uint32_t m_motivation); // same as TransferMotivation but transfer in all directions.
private:
	void TransferMotivation(MotivationTransferNeuron* neighbour, uint32_t motivation);

	PremotorNeuron *m_premotorNeuron;
	PPh::VectorInt32Math m_pos3D;
	std::array <TransferMotivationArray, 2> m_transferMotivation;
	uint32_t m_transferCentralMotivation[2] = { 0,0 }; // same as m_transferMotivation but transfer in all directions.
	uint32_t m_reinforcement = 0;
	const Neuron *m_reinforcementActivator = 0;
	uint32_t m_CentralMotivationSource[2] = { 0,0 }; // this neuron is the source of motivation. Will transfer it in all directions, but not on itself.
	uint32_t m_axon[2];
};

class ActivatorNeuron : public Neuron
{
public:
	ActivatorNeuron() = default;
	virtual ~ActivatorNeuron() = default;

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ActivatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	void Inhibit(uint32_t) override;
	uint32_t ReadAxon() const override;

private:
	uint32_t m_axon[2];
	uint32_t m_inhibitor[2];
};

class MotivationSourceNeuron : public Neuron
{
public:
	MotivationSourceNeuron() = default;
	virtual ~MotivationSourceNeuron() = default;

	void InitExplicit(Neuron *activatorNeuron, MotivationTransferNeuron *motivationTransferNeuron, uint32_t motivation);

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotivationSourceNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;

private:

	MotivationTransferNeuron *m_motivationTransferNeuron;
	uint32_t m_motivation = 0;
	Neuron *m_activatorNeuron;
};

class HungerActivatorNeuron : public Neuron
{
public:
	HungerActivatorNeuron() = default;
	virtual ~HungerActivatorNeuron() = default;

	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::HungerActivatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void InitExplicit(Neuron *newnessActivatorNeuron, MotivationTransferNeuron *motivationTransferNeuron);

	void Tick() override;
	void CrumbEaten();
	bool IsHunger() const;

private:
	Neuron *m_newnessActivatorNeuron;
	MotivationTransferNeuron *m_centralMotivationTransferNeuron;
	uint32_t m_curQuants = 0;
	const uint32_t m_activateQuants = SECOND_IN_QUANTS*30;
	uint32_t m_eatenCrumbNum = 0;
};
